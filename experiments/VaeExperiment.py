import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from loss_functions.nt_xent import NTXentLoss
import os
import shutil
import sys
import pickle
from datetime import datetime

from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from networks.beta_vae import BetaVAE
from networks.vanilla_vae import VanillaVAE

apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class VAEExperiment(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config['save_dir'], self.config["vae_mode"], 'tensorboard'))
        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])

        split_dir = os.path.join(self.config["base_dir"], "splits.pkl")
        self.data_dir = os.path.join(self.config["base_dir"], "preprocessed")

        with open(split_dir, "rb") as f:
            splits = pickle.load(f)
        k = config["fold"]
        tr_keys = splits[k]['train'] + splits[k]['val']
        self.val_keys = splits[k]['train']
        self.train_loader = NumpyDataSet(self.data_dir, target_size=self.config["img_size"], batch_size=self.config["batch_size"],
                                         keys=tr_keys, do_reshuffle=True)
        self.val_loader = NumpyDataSet(self.data_dir, target_size=self.config["img_size"], batch_size=self.config["val_batch_size"],
                                         keys=self.val_keys, do_reshuffle=True)

        print(len(self.train_loader))
        if self.config["vae_mode"] == "beta":
            self.model = BetaVAE(in_channels=1, latent_dim=256).to(self.device)
        elif self.config["vae_mode"] == "base":
            self.model = VanillaVAE(in_channels=1, latent_dim=256).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        self.save_folder = os.path.join("output_experiment", "infer_vae" + str(datetime.now())[0:16])
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self):

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,  gamma=0.95, last_epoch=-1)

        for epoch_counter in range(self.config['epochs']):
            print("=====Training Epoch: %d =====" % epoch_counter)
            for i, data_batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                img = data_batch['data'][0].float().to(self.device)
                label = data_batch['seg'][0].long().to(self.device)

                results = self.model(img, labels=label)
                train_loss = self.model.loss_function(*results,
                                                      M_N=self.config['batch_size'] / len(self.train_loader),
                                                      optimizer_idx=0,
                                                      batch_idx=i)

                loss = train_loss["loss"]
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(self.train_loader),
                                                                          loss=loss.item()))

                loss.backward()
                self.optimizer.step()
                n_iter += 1

            torch.save(self.model.state_dict(), os.path.join(self.config['save_dir'], self.config["vae_mode"],
                                                             'b_{}_f{}_vae.pth'.format(self.config["batch_size"],
                                                                                            self.config["fold"])))

            """
            print("===== Validation =====")
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(self.val_loader)
                print("Val:[{0}] loss: {loss:.4f}".format(epoch_counter, loss=valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.config['save_dir'],
                                                                     'b_{}_model.pth'.format(self.config["batch_size"])))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            """

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, valid_loader):

        # validation steps
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in valid_loader:
                xis = xis['data'][0].float().to(self.device)
                xjs = xjs['data'][0].float().to(self.device)

                loss = self._step(self.model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        return valid_loss

    def load_checkpoint(self):
        if self.config["saved_model_path"] is None:
            print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            exit(0)
        else:
            checkpoint = torch.load(self.config["saved_model_path"])
            if "model" not in checkpoint.keys():
                state_dict = checkpoint
            else:
                state_dict = checkpoint["model"]
            self.model.load_state_dict(state_dict, strict=False)
            print("checkpoint state dict:", state_dict.keys())
            print("model state dict:", self.model.state_dict().keys())

    def infer(self):
        self.load_checkpoint()

        with torch.no_grad():
            for k in range(len(self.val_keys)):
                key = self.val_keys[k:k+1]
                data_loader = NumpyDataSet(self.data_dir, target_size=self.config["img_size"],
                                             batch_size=1, keys=key, do_reshuffle=False, mode="test")
                feature_map = []
                reconstruct_img = []
                prediction = []
                for (i, data_batch) in enumerate(data_loader):
                    data = data_batch['data'][0].float().to(self.device)
                    labels = data_batch['seg'][0].long().to(self.device)
                    slice_idx = data_batch['slice_idxs']

                    features = self.model(data, infer=True)
                    # print(output.shape, labels.shape)

                    # features = F.normalize(features, p=2, dim=1)
                    img = features[0]
                    mu = features[2]
                    logvar = features[3]
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    features = mu
                    features = F.normalize(features, p=2, dim=1)
                    for j in range(features.shape[0]):
                        feature_map.append(features[j].cpu().numpy())
                        reconstruct_img.append(img[j].cpu().numpy())
                    # print(features.shape, labels.shape)

                feature_map = np.stack(feature_map)
                self.save_data(feature_map, key, 'features')
                # self.save_data(reconstruct_img, key, "reconstruct")

    def save_data(self, data, key, mode):

        if not os.path.exists(os.path.join(self.save_folder, mode)):
            os.mkdir(os.path.join(self.save_folder, mode))

        save_path = os.path.join(self.save_folder, mode + '_' + key[0])
        np.save(save_path, data)
