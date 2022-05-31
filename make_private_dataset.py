import cv2
import numpy as np
import matplotlib.pyplot as plt
from hyperpose import Config, Model, Dataset
from hyperpose.Model.openpose.define import CocoColor
from pathlib import Path
import os
from hyperpose.Dataset import imread_rgb_float, imwrite_rgb_float
from skimage import color, data, restoration
import json
import tensorflow as tf
import hyperpose.Model.optics.optics_utils as optics_utils


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    idname = parts[-2]
    fname = parts[-1]

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, idname, fname


# "Ori_priv_model" # "private_OPPS_VGG19_OSIRIM_fine_tune_20" # "low_resolution"  # "slow_train_fine_tune_22"
priv_model_name = "WithFaceRecog_NO_Optics_Loss"
data_path = "/scr/ms1m_align_112/imgs"
# L1 -> Original approach sent to CVPR
# L2 -> L1 - Face Keypoints
# L3 -> Body - Face recognition
priv_data_path = "/scr/ms1m_align_112/L3_priv_imgs_6x/"
Config.set_model_name(priv_model_name)
Config.set_model_type(Config.MODEL.Openpose)
Config.set_model_backbone(Config.BACKBONE["Vgg19"])
Config.set_privacy_preserving_model(True)
Config.set_config_optics_file("optics_cfg.json")
config = Config.get_config()

# get and load model
model = Model.get_model(config)
weight_path = f"{config.model.model_dir}/newest_model.npz"  # newest_model.npz"  # partial_results/model_25000.npz"
model.load_weights(weight_path)

list_ds = tf.data.Dataset.list_files(os.path.join(data_path, "*/*"))

images_ds = list_ds.map(parse_image).batch(200)

image_dims = [112, 112]
aug_dims = [112 * 4, 112 * 4]
k = 0
for image, idname, fname in images_ds:
    image = tf.pad(image, [[0, 0], [56 * 2, 56 * 2], [56 * 2, 56 * 2], [0, 0]], "SYMMETRIC")
    x_sensor, outputs = model.infer(image)
    x_sensor = tf.image.resize(x_sensor, aug_dims)
    x_sensor = tf.image.central_crop(x_sensor, 0.33)
    x_sensor = tf.image.resize(x_sensor, image_dims)  # for x4 size
    x_sensor = tf.image.convert_image_dtype(x_sensor, dtype=tf.uint8)
    for ix in range(x_sensor.shape[0]):
        tmp = tf.io.encode_jpeg(x_sensor[ix], 'rgb')
        save_path = os.path.join(priv_data_path, idname[ix].numpy().decode())
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tf.io.write_file(os.path.join(save_path, fname[ix].numpy().decode()), tmp)

    k += 1
    if k % 500:
        print("Processing batch {}... \n".format(k))
