import argparse
import os

import torch

import data
import models
from util import util


class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.isTrain = True

    def initialize(self, parser):
        parser.add_argument(
            "--dataroot",
            type=str,
            default="./dataset/test_image/",
            help="path to images (should have subfolders train(test)/img/video_name/frame_files)",
        )
        parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
        parser.add_argument("--loadSize", type=int, default=286, help="scale images to this size")
        parser.add_argument(
            "--display_winsize", type=int, default=256, help="display window size for both visdom and HTML"
        )
        parser.add_argument("--fineSize", type=int, default=256, help="then crop to this size")
        parser.add_argument("--input_nc", type=int, default=3, help="# of input image channels")
        parser.add_argument("--output_nc", type=int, default=3, help="# of output image channels")
        parser.add_argument("--ngf", type=int, default=64, help="# of gen filters in first conv layer")
        parser.add_argument("--ndf", type=int, default=64, help="# of discrim filters in first conv layer")
        parser.add_argument("--netD", type=str, default="basic", help="selects model to use for netD")
        parser.add_argument("--n_layers_D", type=int, default=3, help="only used if which_model_netD==n_layers")
        parser.add_argument("--netG", type=str, default="unet256", help="selects model to use for netG")
        parser.add_argument("--gpu_ids", type=str, default="-1", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        parser.add_argument(
            "--name",
            type=str,
            required=True,
            help="name of the experiment. It decides where to store samples and models",
        )
        parser.add_argument("--dataset_mode", type=str, default="affineGAN", help="chooses how datasets are loaded")
        parser.add_argument("--model", type=str, default="affineGAN", help="chooses which model to use")
        parser.add_argument(
            "--epoch",
            type=str,
            default="latest",
            help="which epoch to load? set to latest to use " "latest cached model",
        )
        parser.add_argument("--num_threads", default=1, type=int, help="# threads for loading data")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")
        parser.add_argument(
            "--norm", type=str, default="instance", help="instance normalization or batch normalization"
        )
        parser.add_argument(
            "--serial_batches",
            action="store_true",
            help="if true, takes images in order to make batches, otherwise takes them randomly",
        )
        parser.add_argument("--no_dropout", action="store_true", help="no dropout for the generator")
        parser.add_argument(
            "--max_dataset_size",
            type=int,
            default=float("inf"),
            help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.",
        )
        parser.add_argument(
            "--resize_or_crop",
            type=str,
            default="resize_and_crop",
            help="scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]",
        )
        parser.add_argument(
            "--no_flip", action="store_true", help="if specified, do not flip the images for data augmentation"
        )
        parser.add_argument(
            "--init_type", type=str, default="normal", help="network initialization [normal|xavier|kaiming|orthogonal]"
        )
        parser.add_argument(
            "--init_gain", type=float, default=0.02, help="scaling factor for normal, xavier and orthogonal."
        )
        parser.add_argument("--verbose", action="store_true", help="if specified, print more debugging information")
        parser.add_argument(
            "--suffix",
            default="",
            type=str,
            help="customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}",
        )

        # projects specific arguments
        parser.add_argument("--no_patch", action="store_true", help="if specified, do not use local mouth patches")
        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def gather_options(self):
        # initialize parser with basic options
        parser = None

        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        if len(opt.gpu_ids) > 0:
            if torch.cuda.is_available():
                torch.cuda.set_device(opt.gpu_ids[0])
            else:
                print("Warning: GPU is not available, CPU is using")

        self.opt = opt
        return self.opt
