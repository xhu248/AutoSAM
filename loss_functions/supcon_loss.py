from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """modified supcon loss for segmentation application, the main difference is that the label for different view
    could be different if after spatial transformation"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        # input features shape: [bsz, v, c, w, h]
        # input labels shape: [bsz, v, w, h]
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # of size (bsz*v, c, h, w)

        kernels = contrast_feature.permute(0, 2, 3, 1)
        kernels = kernels.reshape(-1, contrast_feature.shape[1], 1, 1)
        # kernels = kernels[non_background_idx]
        logits = torch.div(F.conv2d(contrast_feature, kernels), self.temperature)  # of size (bsz*v, bsz*v*h*w, h, w)
        logits = logits.permute(1, 0, 2, 3)
        logits = logits.reshape(logits.shape[0], -1)

        if labels is not None:
            labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

            bg_bool = torch.eq(labels.squeeze().cpu(), torch.zeros(labels.squeeze().shape))
            non_bg_bool = ~ bg_bool
            non_bg_bool = non_bg_bool.int().to(device)
        else:
            mask = torch.eye(logits.shape[0]//contrast_count).float().to(device)
            mask = mask.repeat(contrast_count, contrast_count)
            # print(mask.shape)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        if labels is not None:
            # only consider the contrastive loss for non-background pixel
            loss = (loss * non_bg_bool).sum() / (non_bg_bool.sum())
        else:
            loss = loss.mean()
        return loss


class SupConSegLoss(nn.Module):
    # TODO: only support batch size = 1
    def __init__(self, temperature=0.7):
        super(SupConSegLoss, self).__init__()
        self.temp = temperature
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def forward(self, features, labels=None):
        # input features: [bsz, c, h ,w], h & w are the image size
        shape = features.shape
        img_size = shape[-1]
        if labels is not None:
            f1, f2 = torch.split(features, [1, 1], dim=1)
            features = torch.cat([f1.squeeze(1), f2.squeeze(1)], dim=0)
            l1, l2 = torch.split(labels, [1, 1], dim=1)
            labels = torch.cat([l1.squeeze(1), l2.squeeze(1)], dim=0)
            # features = features.squeeze(dim=1)
            # labels = labels.squeeze(dim=1)
            bsz = features.shape[0]
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                for i in range(img_size):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    for j in range(img_size):
                        x = features[b:b + 1, :, i:i + 1, j:j + 1]  # [1,c, 1, 1, 1]
                        x_label = labels[b, i, j] + 1  # avoid cases when label=0
                        if x_label == 1:  # ignore background
                            continue
                        cos_dst = F.conv2d(features, x)  # [2b, 1, 512, 512]
                        cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                        # print("cos_dst:", cos_dst.max(), cos_dst.min())
                        self_contrast_dst = torch.div((x * x).sum(), self.temp)

                        mask = labels + 1
                        mask[mask != x_label] = 0
                        # if mask.sum() < 5:
                        #    print("Not enough same label pixel")
                        #    continue
                        mask = torch.div(mask, x_label)
                        numerator = (mask * cos_dst).sum() - self_contrast_dst
                        denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                        # print("denominator:", denominator.item())
                        # print("numerator:", numerator.max(), numerator.min())
                        loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                        if loss_tmp != loss_tmp:
                            print(numerator.item(), denominator.item())

                        loss.append(loss_tmp)
            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss

        else:
            bsz = features.shape[0]
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                tmp_feature = features[b]
                for n in range(tmp_feature.shape[0]):
                    for i in range(img_size):
                        # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                        for j in range(img_size):
                            x = tmp_feature[n:n+1, :, i:i + 1, j:j + 1]  # [c, 1, 1, 1]
                            cos_dst = F.conv2d(tmp_feature, x)  # [2b, 1, 512, 512]
                            cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                            # print("cos_dst:", cos_dst.max(), cos_dst.min())
                            self_contrast_dst = torch.div((x * x).sum(), self.temp)

                            mask = torch.zeros((tmp_feature.shape[0], tmp_feature.shape[2], tmp_feature.shape[3]),
                                               device=self.device)
                            mask[0:tmp_feature.shape[0], i, j] = 1
                            numerator = (mask * cos_dst).sum() - self_contrast_dst
                            denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                            # print("numerator:", numerator.max(), numerator.min())
                            loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                            if loss_tmp != loss_tmp:
                                print(numerator.item(), denominator.item())

                            loss.append(loss_tmp)

            loss = torch.stack(loss).mean()
            return loss


class LocalConLoss(nn.Module):
    def __init__(self, temperature=0.7, stride=4, mode='stride'):
        super(LocalConLoss, self).__init__()
        self.temp = temperature
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=self.temp)
        self.stride = stride
        self.mode = mode

    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        if self.mode == 'local':
            self.stride = (features.shape[3] - 1) // 3
        features = features[:, :, :, ::self.stride, ::self.stride]  # resample feature maps to reduce memory consumption and running time
        shape = features.shape
        img_size = shape[-1]
        if labels is not None:
            labels = labels[:, :, ::self.stride, ::self.stride]
            if labels.sum() == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss

            loss = self.supconloss(features, labels)
            """
            f1, f2 = torch.split(features, [1, 1], dim=1)
            features = torch.cat([f1.squeeze(1), f2.squeeze(1)], dim=0)
            l1, l2 = torch.split(labels, [1, 1], dim=1)
            labels = torch.cat([l1.squeeze(1), l2.squeeze(1)], dim=0)
            bsz = features.shape[0]
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                for i in range(img_size):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    for j in range(img_size):
                        x = features[b:b + 1, :, i:i + 1, j:j + 1]  # [c, 1, 1, 1]
                        x_label = labels[b, i, j] + 1  # avoid cases when label=0
                        if x_label == 1:  # ignore background
                            continue
                        cos_dst = F.conv2d(features, x)  # [2b, 1, 512, 512]
                        cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                        self_contrast_dst = torch.div((x * x).sum(), self.temp)

                        mask = labels + 1
                        mask[mask != x_label] = 0
                        mask = torch.div(mask, x_label)
                        numerator = (mask * cos_dst).sum() - self_contrast_dst
                        denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                        # print("denominator:", denominator.item())
                        # print("numerator:", numerator.max(), numerator.min())
                        loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                        if loss_tmp != loss_tmp:
                            print(numerator.item(), denominator.item())

                        loss.append(loss_tmp)

            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            """
            return loss
        else:
            bsz = features.shape[0]
            loss = self.supconloss(features)

            """
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                tmp_feature = features[b]
                for n in range(tmp_feature.shape[0]):
                    for i in range(img_size):
                        # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                        for j in range(img_size):
                            x = tmp_feature[n:n+1, :, i:i + 1, j:j + 1]  # [c, 1, 1, 1]
                            cos_dst = F.conv2d(tmp_feature, x)  # [2b, 1, 512, 512]
                            cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                            # print("cos_dst:", cos_dst.max(), cos_dst.min())
                            self_contrast_dst = torch.div((x * x).sum(), self.temp)

                            mask = torch.zeros((tmp_feature.shape[0], tmp_feature.shape[2], tmp_feature.shape[3]),
                                               device=self.device)
                            mask[0:tmp_feature.shape[0], i, j] = 1
                            numerator = (mask * cos_dst).sum() - self_contrast_dst
                            denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                            # print("numerator:", numerator.max(), numerator.min())
                            loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                            if loss_tmp != loss_tmp:
                                print(numerator.item(), denominator.item())

                            loss.append(loss_tmp)

            loss = torch.stack(loss).mean()
            """
            return loss


class BlockConLoss(nn.Module):
    def __init__(self, temperature=0.7, block_size=32):
        super(BlockConLoss, self).__init__()
        self.block_size = block_size
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=temperature)

    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        shape = features.shape
        img_size = shape[-1]
        div_num = img_size // self.block_size
        if labels is not None:
            loss = []
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]
                    block_labels = labels[:,:, i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]

                    if block_labels.sum() == 0:
                        continue

                    tmp_loss = self.supconloss(block_features, block_labels)

                    loss.append(tmp_loss)

            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss

        else:
            loss = []
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i * self.block_size:(i + 1) * self.block_size,
                                     j * self.block_size:(j + 1) * self.block_size]

                    tmp_loss = self.supconloss(block_features)

                    loss.append(tmp_loss)

            loss = torch.stack(loss).mean()
            return loss