import os
from .define import MODEL, DATA, TRAIN, BACKBONE
from easydict import EasyDict as edict

# model configuration
model = edict()
# number of keypoints + 1 for background
model.n_pos = 19
model.num_channels = 128
# input size during training , 240
model.hin = 368
model.win = 432
# output size during training (default 46)
model.hout = 46
model.wout = 54
model.model_type = MODEL.Openpose
model.model_name = "default_name"
model.model_backbone = BACKBONE.Default
model.data_format = "channels_last"
# save directory
model.model_dir = f"./save_dir/{model.model_name}/model_dir"

# train configuration
train = edict()
train.batch_size = 8
train.save_interval = 2000
# total number of step
train.n_step = 50000  # 1000000
# initial learning rate  
train.lr_init = 1e-4
# evey number of step to decay lr
train.lr_decay_every_step = 5000  # 136120
# decay lr factor
train.lr_decay_factor = 0.666
train.weight_decay_factor = 1e-4
train.train_type = TRAIN.Single_train
train.vis_dir = f"./save_dir/{model.model_name}/train_vis_dir"
train.pretrained_model = './pretrained_models/hyperpose/new_opps.npz'
train.two_face_rec_models = False
# eval configuration
eval = edict()
eval.batch_size = 22
eval.vis_dir = f"./save_dir/{model.model_name}/eval_vis_dir"

# data configuration
data = edict()
data.dataset_type = DATA.MSCOCO  # coco, custom, coco_and_custom
data.dataset_version = "2017"  # MSCOCO version 2014 or 2017
data.dataset_path = "./data"
data.dataset_filter = None
data.vis_dir = f"./save_dir/data_vis_dir"
data.train_tfrecord_name = "coco_pose_data_with_faces.tfrecord"
# log configuration
log = edict()
log.log_interval = 100
log.log_path = f"./save_dir/{model.model_name}/log.txt"
log.log_path = f"./save_dir/{model.model_name}/log.txt"
log.test_img1 = "./test_images/054820377.jpg"
log.test_img2 = "./test_images/056115021.jpg"
log.test_img3 = "./test_images/056407403.jpg"
