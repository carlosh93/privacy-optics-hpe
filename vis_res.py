import cv2
import numpy as np
import matplotlib.pyplot as plt
from hyperpose import Config, Model, Dataset
from hyperpose.Model.openpose.define import CocoColor
from pathlib import Path
from hyperpose.Dataset import imread_rgb_float, imwrite_rgb_float
from skimage import color, data, restoration
import json
import tensorflow as tf
import hyperpose.Model.optics.optics_utils as optics_utils

priv_model_name = "private_OPPS_VGG19_OSIRIM_fine_tune_20" # "low_resolution"  # "slow_train_fine_tune_22"  # fine_tune_20
spec = ""
filename = "sylvester.jpg"

Config.set_model_name(priv_model_name)
Config.set_model_type(Config.MODEL.Openpose)
Config.set_model_backbone(Config.BACKBONE["Vgg19"])
Config.set_privacy_preserving_model(True)
Config.set_config_optics_file("optics_cfg.json")
config = Config.get_config()

# get and load model
model = Model.get_model(config)
weight_path = f"{config.model.model_dir}/newest_model.npz" #newest_model.npz"  # partial_results/model_20000.npz"
model.load_weights(weight_path)

ori_image = cv2.cvtColor(cv2.imread("./test_images/" + filename), cv2.COLOR_BGR2RGB)
ori_image = ori_image.astype(np.float32) / 255.0
# if (model.pose_model.data_format == "channels_first"):
#    input_image = np.transpose(input_image, [2, 0, 1])

x_sensor, outputs = model.infer(ori_image[np.newaxis, :, :, :])

conf_map = outputs[0]
paf_map = outputs[1]
# get visualize function, which is able to get visualized part and limb heatmap image from inferred heatmaps
visualize = Model.get_visualize(Config.MODEL.Openpose)
# draw all detected skeletons
output_img = x_sensor[0].numpy().copy()
input_img = x_sensor[0].numpy().copy()
img_h, img_w, img_c = input_img.shape
x_sensor = np.transpose(x_sensor[0].numpy(), [2, 0, 1])
vis_parts_heatmap, vis_limbs_heatmap = visualize(x_sensor, conf_map[0], paf_map[0], save_tofile=False, )

# get postprocess function, which is able to get humans that contains assembled detected parts from inferred heatmaps
postprocess = Model.get_postprocess(Config.MODEL.Openpose)
humans = postprocess(conf_map[0], paf_map[0], img_h, img_w, model.pose_model.parts, model.pose_model.limbs,
                     model.pose_model.data_format, (np.array(CocoColor)/255).tolist())

for human in humans:
    output_img = human.draw_human(output_img)

Path("vis_results").mkdir(parents=True, exist_ok=True)
Path("vis_results/near_"+filename.split('.')[0]).mkdir(parents=True, exist_ok=True)
plt.imsave("vis_results/near_"+filename.split('.')[0]+"/rgb_img.png", cv2.resize(ori_image, (img_h, img_w)))
plt.imsave("vis_results/near_"+filename.split('.')[0]+"/sensor_img_priv.png", cv2.resize(input_img, (img_h, img_w)))
plt.imsave("vis_results/near_"+filename.split('.')[0]+"/res_img_priv.png", cv2.resize(output_img, (img_h, img_w)))

# Original OpenPose Model

Config.set_model_name("openpose")
Config.set_model_type(Config.MODEL.Openpose)
Config.set_model_backbone(Config.BACKBONE["Vgg19"])
Config.set_privacy_preserving_model(False)
config = Config.get_config()

# get and load model
model = Model.get_model(config)
model.load_weights(config.train.pretrained_model)
conf_map, paf_map = model.infer(ori_image[np.newaxis, :, :, :])

# get visualize function, which is able to get visualized part and limb heatmap image from inferred heatmaps
visualize = Model.get_visualize(Config.MODEL.Openpose)
# draw all detected skeletons
output_img = ori_image.copy()
input_img = ori_image.copy()
img_h2, img_w2, _ = input_img.shape
input_img = np.transpose(ori_image, [2, 0, 1])
vis_parts_heatmap, vis_limbs_heatmap = visualize(input_img, conf_map[0], paf_map[0], save_tofile=False, )

# get postprocess function, which is able to get humans that contains assembled detected parts from inferred heatmaps
postprocess = Model.get_postprocess(Config.MODEL.Openpose)
humans = postprocess(conf_map[0], paf_map[0], img_h2, img_w2, model.parts, model.limbs,
                     model.data_format, (np.array(CocoColor)/255).tolist())

for human in humans:
    output_img = human.draw_human(output_img)

plt.imsave("vis_results/near_"+filename.split('.')[0]+"/sensor_img.png", cv2.resize(ori_image, (img_h, img_w)))
plt.imsave("vis_results/near_"+filename.split('.')[0]+"/res_img.png", cv2.resize(output_img, (img_h, img_w)))
# if you want to visualize all the images in one plot:
# show image,part heatmap,limb heatmap and detected image
# here we use 'transpose' because our data_format is 'channels_first'
"""
fig = plt.figure(figsize=(8, 8))
# origin image
origin_fig = fig.add_subplot(2, 2, 1)
origin_fig.set_title("origin image")
origin_fig.imshow(input_img)
# parts heatmap
parts_fig = fig.add_subplot(2, 2, 2)
parts_fig.set_title("parts heatmap")
parts_fig.imshow(vis_parts_heatmap)
# limbs heatmap
limbs_fig = fig.add_subplot(2, 2, 3)
limbs_fig.set_title("limbs heatmap")
limbs_fig.imshow(vis_limbs_heatmap)
# detected results
result_fig = fig.add_subplot(2, 2, 4)
result_fig.set_title("detect result")
result_fig.imshow(output_img)
# save fig
plt.savefig("./sample_custome_infer.png")
plt.close()
"""