from experiments.simclr_experiment import SimCLR
from experiments.contrast_experiment import ContrastExperiment
from experiments.ByolExperiment import BYOLExperiment
import yaml
import argparse


def parse_option():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--dataset", type=str, default="hippo")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-f", "--fold", type=int, default=1)
    parser.add_argument("--contrast_mode", type=str, default="simclr")
    parser.add_argument("--pseudo_file", type=str, default="pseudo_label.pkl")
    parser.add_argument('--src_dir', type=str, default=None, help='path to splits file')
    parser.add_argument('--data_dir', type=str, default=None, help='path to datafolder')
    parser.add_argument("--tr_size", type=int, default=1)
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_option()
    args.distributed = False
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.dataset == "mmwhs":
        config["img_size"] = 256
        config["base_dir"] = "data/mmwhs/"
        config["save_dir"] = "save/" + args.contrast_mode + "/mmwhs"
    elif args.dataset == "synapse":
        config["save_dir"] = "save/" + args.contrast_mode + "synapse"
        config["img_size"] = 256
    elif args.dataset == "ACDC":
        config["save_dir"] = "save/" + args.contrast_mode + "_ACDC"
        config["img_size"] = 224

    config['batch_size'] = args.batch_size
    config["fold"] = args.fold
    config['epochs'] = args.epoch
    config["pseudo_label_file"] = args.pseudo_file
    print(config)

    if args.contrast_mode == "simclr":
        simclr = SimCLR(config, args)
        simclr.train()
    elif "contrast" in args.contrast_mode:
        contrast = ContrastExperiment(config)
        contrast.train()
    elif args.contrast_mode == "byol":
        byol = BYOLExperiment(config, args)
        byol.train()
