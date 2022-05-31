#!/usr/bin/env python3

import os
import cv2
import sys
import math
import json
import glob
import argparse
import matplotlib
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from hyperpose import Config, Model, Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastPose.')
    parser.add_argument("--model_type",
                        type=str,
                        default="Openpose",
                        help="human pose estimation model type, available options: Openpose, LightweightOpenpose ,MobilenetThinOpenpose, PoseProposal")
    parser.add_argument("--model_backbone",
                        type=str,
                        default="Vgg19",
                        help="model backbone, available options: Mobilenet, Vggtiny, Vgg19, Resnet18, Resnet50")
    parser.add_argument("--model_name",
                        type=str,
                        default="private_OPPS",
                        help="model name,to distinguish model and determine model dir")
    parser.add_argument("--dataset_type",
                        type=str,
                        default="MSCOCO",
                        help="dataset name,to determine which dataset to use, available options: MSCOCO, MPII ")
    parser.add_argument("--dataset_version",
                        type=str,
                        default="2017",
                        help="dataset version, only use for MSCOCO and available for version 2014 and 2017")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="data",
                        help="dataset path,to determine the path to load the dataset")
    parser.add_argument("--get_data_from_TFrecord",
                        type=bool,
                        default=True,
                        help="Should get data from TFrecord or from raw data")
    parser.add_argument('--train_type',
                        type=str,
                        default="Single_train",
                        help='train type, available options: Single_train, Parallel_train')
    parser.add_argument('--kf_optimizer',
                        type=str,
                        default='Sync_avg',
                        help='kung fu parallel optimizor,available options: Sync_sgd, Sync_avg, Pair_avg')
    parser.add_argument('--use_official_dataset',
                        type=int,
                        default=1,
                        help='whether to use official dataset, could be used when only user data is needed')
    parser.add_argument('--useradd_data_path',
                        type=str,
                        default=None,
                        help='path to user data directory where contains images folder and annotation json file')
    parser.add_argument('--domainadapt_data_path',
                        type=str,
                        default=None,
                        help='path to user data directory where contains images for domain adaptation')
    parser.add_argument('--optim_type',
                        type=str,
                        default="Adam",
                        help='optimizer type used for training')
    parser.add_argument('--privacy_model',
                        type=bool,
                        default=True,
                        help='Make a privacy-preserving openpose model')
    parser.add_argument('--optics_config_file',
                        type=str,
                        default="optics_cfg.json",
                        help='Make a privacy-preserving openpose model')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,  # 2e-5 # 1e-4,
                        help='learning rate')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch Size')
    parser.add_argument('--log_interval',
                        type=int,
                        default=500,
                        help='log frequency')
    parser.add_argument('--save_interval',
                        type=int,
                        default=5000,
                        help='log frequency')

    args = parser.parse_args()
    # config model
    Config.set_model_name(args.model_name)
    Config.set_model_type(Config.MODEL[args.model_type])
    Config.set_model_backbone(Config.BACKBONE[args.model_backbone])
    Config.set_privacy_preserving_model(args.privacy_model)
    Config.set_get_data_from_tfrecord(args.get_data_from_TFrecord)
    Config.set_config_optics_file(args.optics_config_file)
    Config.set_batch_size(args.batch_size)
    Config.set_log_interval(args.log_interval)
    Config.set_save_interval(args.save_interval)
    # config train
    Config.set_train_type(Config.TRAIN[args.train_type])
    Config.set_learning_rate(args.learning_rate)
    Config.set_optim_type(Config.OPTIM[args.optim_type])
    Config.set_kungfu_option(Config.KUNGFU[args.kf_optimizer])
    # config dataset
    print(f"test enabling official dataset:{args.use_official_dataset}")
    Config.set_official_dataset(args.use_official_dataset)
    Config.set_dataset_type(Config.DATA[args.dataset_type])
    Config.set_dataset_path(args.dataset_path)
    Config.set_dataset_version(args.dataset_version)
    # sample add user data to train
    if (args.useradd_data_path != None):
        useradd_train_image_paths = []
        useradd_train_targets = []
        image_dir = os.path.join(args.useradd_data_path, "images")
        anno_path = os.path.join(args.useradd_data_path, "anno.json")
        # generate image paths and targets
        anno_json = json.load(open(anno_path, mode="r"))
        for image_path in anno_json["annotations"].keys():
            anno = anno_json["annotations"][image_path]
            useradd_train_image_paths.append(os.path.join(image_dir, image_path))
            useradd_train_targets.append({
                "kpt": anno["keypoints"],
                "mask": None,
                "bbx": anno["bbox"],
                "labeled": 1
            })
        Config.set_useradd_data(useradd_train_image_paths, useradd_train_targets, useradd_scale_rate=1)
    # sample use domain adaptation to train:
    if (args.domainadapt_data_path != None):
        domainadapt_image_paths = glob.glob(os.path.join(args.domainadapt_data_path, "images", "*"))
        Config.set_domainadapt_dataset(domainadapt_train_img_paths=domainadapt_image_paths, domainadapt_scale_rate=1)
    # train
    config = Config.get_config()
    model = Model.get_model(config)
    train = Model.get_train(config)
    dataset = Dataset.get_dataset(config, args.get_data_from_TFrecord)
    # dataset.save_train_tfrecord_dataset()
    train(model, dataset)
