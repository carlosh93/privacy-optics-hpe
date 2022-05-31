#!/usr/bin/env python3

import math
import multiprocessing
import os
import cv2
import time
import sys
import json
import numpy as np
import matplotlib
import wandb

matplotlib.use('Agg')
import tensorflow as tf
import tensorlayer as tl
from pycocotools.coco import maskUtils
import _pickle as cPickle
from functools import partial
from .utils import tf_repeat, get_heatmap, get_vectormap, draw_results
from .utils import get_parts, get_limbs, get_flip_list
from ..common import log, KUNGFU, MODEL, get_optim, init_log
from ..domainadapt import get_discriminator
import globals as globals_vars
from train_utils import *
# from arcface_model.models import ArcFaceModel
import yaml

face_rec_loss_fn = tf.keras.losses.CosineSimilarity()


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def regulize_loss(target_model, weight_decay_factor):
    re_loss = 0
    regularizer = tf.keras.regularizers.l2(l=weight_decay_factor)
    for trainable_weight in target_model.trainable_weights[1:target_model.last_tuned_layer]:
        re_loss += regularizer(trainable_weight)
    return re_loss, regularizer(target_model.trainable_weights[0])


def _data_aug_fn(image, ground_truth, hin, hout, win, wout, parts, limbs, flip_list=None, data_format="channels_first"):
    """Data augmentation function."""
    # restore data
    concat_dim = 0 if data_format == "channels_first" else -1
    ground_truth = cPickle.loads(ground_truth.numpy())
    image = image.numpy()
    annos = ground_truth["kpt"]
    labeled = ground_truth["labeled"]
    mask = ground_truth["mask"]
    try:
        face_boxes = tf.reshape(ground_truth["face_anns"]["boxes"], [-1, 2]).numpy()
        # face_img = tf.image.convert_image_dtype(ground_truth["face_anns"]["faces"], tf.float32)
        # faces = face_img / tf.reduce_max(face_img)
    except:
        # Annotations doesn't exists
        face_boxes = None
        # faces = tf.zeros(shape=(1, 112, 112, 3))

    # decode mask
    h_mask, w_mask, _ = np.shape(image)
    mask_miss = np.ones((h_mask, w_mask), dtype=np.uint8)
    if (mask != None):
        for seg in mask:
            bin_mask = maskUtils.decode(seg)
            bin_mask = np.logical_not(bin_mask)
            if (bin_mask.shape != mask_miss.shape):
                print(f"test error mask shape mask_miss:{mask_miss.shape} bin_mask:{bin_mask.shape}")
            else:
                mask_miss = np.bitwise_and(mask_miss, bin_mask)

    # get transform matrix
    # uncomment the lines below to get data augmentation
    # M_rotate = tl.prepro.affine_rotation_matrix(angle=(-30, 30))  # original paper: -40~40
    # M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 0.8))  # original paper: 0.5~1.1
    # M_combined = M_rotate.dot(M_zoom)
    h, w, _ = image.shape
    # transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)

    # apply data augmentation
    # image = tl.prepro.affine_transform_cv2(image, transform_matrix)
    # mask_miss = tl.prepro.affine_transform_cv2(mask_miss, transform_matrix, border_mode='replicate')
    # annos = tl.prepro.affine_transform_keypoints(annos, transform_matrix)
    # temply ignore flip augmentation
    '''
    if(flip_list!=None):
        image, annos, mask_miss = tl.prepro.keypoint_random_flip(image,annos, mask_miss, prob=0.5, flip_list=flip_list)
    '''

    # Transform face annotations
    num_faces = 5
    if face_boxes is not None:
        face_boxes = face_boxes[:, [1, 0]]
        face_boxes[:, 0] *= w
        face_boxes[:, 1] *= h
        annos.append(face_boxes)

    image, annos, mask_miss = tl.prepro.keypoint_resize_random_crop(image, annos, mask_miss, size=(hin, win))  # hao add

    result_face_boxes = np.zeros(shape=(num_faces, 4))
    if face_boxes is not None:
        # Transform Face annotations back
        face_boxes = np.array(annos[-1]).astype(np.float32)
        annos.pop()
        if np.sum(face_boxes < 0) == 0:
            face_boxes[:, 0] /= win
            face_boxes[:, 1] /= hin
            face_boxes = face_boxes[:, [1, 0]]
            face_boxes = np.reshape(face_boxes, (-1, 4))
            result_face_boxes[:min(face_boxes.shape[0], num_faces), :] = face_boxes[
                                                                         :min(face_boxes.shape[0], num_faces), :]

    # generate result which include keypoints heatmap and vectormap
    height, width, _ = image.shape
    heatmap = get_heatmap(annos, height, width, hout, wout, parts, limbs, data_format=data_format)
    vectormap = get_vectormap(annos, height, width, hout, wout, parts, limbs, data_format=data_format)
    resultmap = np.concatenate((heatmap, vectormap), axis=concat_dim)

    image = cv2.resize(image, (win, hin))
    mask_miss = cv2.resize(mask_miss, (win, hin))
    img_mask = mask_miss

    # generate output masked image, result map and maskes
    img_mask = mask_miss.reshape(hin, win, 1)
    image = image * np.repeat(img_mask, 3, 2)
    resultmap = np.array(resultmap, dtype=np.float32)
    mask_miss = np.array(cv2.resize(mask_miss, (wout, hout), interpolation=cv2.INTER_AREA), dtype=np.float32)[:, :,
                np.newaxis]
    if (data_format == "channels_first"):
        image = np.transpose(image, [2, 0, 1])
        mask_miss = np.transpose(mask_miss, [2, 0, 1])
    labeled = np.float32(labeled)
    return image, resultmap, mask_miss, labeled, result_face_boxes


def _map_fn(img_list, annos, data_aug_fn, hin, win, hout, wout, parts, limbs):
    """TF Dataset pipeline."""
    # load data
    # image = tf.io.read_file(img_list)
    image = tf.image.decode_jpeg(img_list, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # data augmentation using affine transform and get paf maps
    image, resultmap, mask, labeled, face_boxes = tf.py_function(data_aug_fn, [image, annos],
                                                                 [tf.float32, tf.float32, tf.float32,
                                                                  tf.float32, tf.float32])
    # data augmentaion using tf
    image = tf.image.random_brightness(image, max_delta=35. / 255.)  # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # lower=0.2, upper=1.8)  caffe 0.3~1.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image, resultmap, mask, labeled, face_boxes


def get_paramed_map_fn(hin, win, hout, wout, parts, limbs, flip_list=None, data_format="channels_first"):
    paramed_data_aug_fn = partial(_data_aug_fn, hin=hin, win=win, hout=hout, wout=wout, parts=parts, limbs=limbs, \
                                  flip_list=flip_list, data_format=data_format)
    paramed_map_fn = partial(_map_fn, data_aug_fn=paramed_data_aug_fn, hin=hin, win=win, hout=hout, wout=wout,
                             parts=parts, limbs=limbs)
    return paramed_map_fn


def calc_face_recognition_loss(image, sensor_image, face_boxes, model_inf, model_train):
    # face_boxes = face_boxes[~np.all(face_boxes == 0, axis=1)]
    # face_boxes = tf.convert_to_tensor(face_boxes)
    face_boxes = tf.reshape(face_boxes, (-1, 4))
    if tf.math.reduce_sum(tf.cast(~tf.reduce_all(face_boxes == 0, axis=1), tf.int8)) == 0:
        return 0

    faces = tf.image.crop_and_resize(image, face_boxes, np.repeat(np.arange(image.shape[0]), 5), [112, 112])
    sensor_faces = tf.image.crop_and_resize(sensor_image, face_boxes, np.repeat(np.arange(image.shape[0]), 5),
                                            [112, 112])

    faces = faces[~tf.reduce_all(face_boxes == 0, axis=1)]
    sensor_faces = sensor_faces[~tf.reduce_all(face_boxes == 0, axis=1)]

    faces_embeddings = tf.math.l2_normalize(model_inf(faces))
    if model_train is not None:
        sensor_faces_embeddings = tf.math.l2_normalize(model_train.forward(sensor_faces))
    else:
        sensor_faces_embeddings = tf.math.l2_normalize(model_inf(sensor_faces))
    return face_rec_loss_fn(faces_embeddings, sensor_faces_embeddings)


def single_train(train_model, dataset, config):
    """Single train pipeline of Openpose class models

    input model and dataset, the train pipeline will start automaticly
    the train pipeline will:
    1.store and restore ckpt in directory ./save_dir/model_name/model_dir
    2.log loss information in directory ./save_dir/model_name/log.txt
    3.visualize model output periodly during training in directory ./save_dir/model_name/train_vis_dir
    the newest model is at path ./save_dir/model_name/model_dir/newest_model.npz

    Parameters
    ----------
    arg1 : tensorlayer.models.MODEL
        a preset or user defined model object, obtained by Model.get_model() function

    arg2 : dataset
        a constructed dataset object, obtained by Dataset.get_dataset() function


    Returns
    -------
    None
    """

    init_log(config)
    # train hyper params
    # dataset params
    n_step = config.train.n_step
    batch_size = config.train.batch_size
    # learning rate params
    lr_init = config.train.lr_init
    lr_decay_factor = config.train.lr_decay_factor
    # lr_decay_steps = [200000, 300000, 360000, 420000, 480000, 540000, 600000, 700000, 800000, 900000]
    lr_decay_steps = [10000, 12000, 15000, 20000,
                      25000]  # [15000, 25000, 28000, 35000, 40000, 45000]  # 10000, 20000, # [30000, 50000, 75000]
    weight_decay_factor = config.train.weight_decay_factor
    # log and checkpoint params
    log_interval = config.log.log_interval
    globals_vars.TF_log_interval = log_interval
    save_interval = config.train.save_interval
    vis_dir = config.train.vis_dir

    # model tensorboard
    # train_summary_writer = train_model.model_summary
    debug = False
    show_freq = 50
    experiment = wandb.init(project='Privacy-HPE', resume='allow', mode="offline" if debug else "online")
    experiment.config.update(dict(show_freq=show_freq, learning_rate=lr_init, n_steps=n_step))

    # model hyper params
    n_pos = train_model.pose_model.n_pos
    hin = train_model.pose_model.hin
    win = train_model.pose_model.win
    hout = train_model.pose_model.hout
    wout = train_model.pose_model.wout
    model_dir = config.model.model_dir
    globals_vars.save_path = os.path.join(model_dir, "partial_results/images/")
    last_tuned_layer = train_model.last_tuned_layer
    set_probe_params(train_model, config)

    pretrain_model_dir = config.pretrain.pretrain_model_dir
    pretrain_model_path = f"{pretrain_model_dir}/newest_{train_model.pose_model.backbone.name}.npz"

    print(f"single training using learning rate:{lr_init} batch_size:{batch_size}")

    if config.data.get_data_from_tfrecord:
        # Get data from TFRecord
        def get_features_description():
            return {
                'img_path': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'raw_img': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'target': tf.io.FixedLenFeature([], tf.string, default_value=''),
            }

        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary obtained by get_features_description.
            return tf.io.parse_single_example(example_proto, get_features_description())

        raw_image_dataset = tf.data.TFRecordDataset(config.data.dataset_path + "/" + config.data.train_tfrecord_name)
        parsed_image_dataset = raw_image_dataset.map(_parse_function)

        # tensorflow data pipeline
        def generator():
            """TF Dataset generator."""
            for coco_example_features in parsed_image_dataset:
                yield coco_example_features['raw_img'].numpy(), coco_example_features['target'].numpy()

        train_dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string))

    else:
        train_dataset = dataset.get_train_dataset()
    # training dataset configure with shuffle,augmentation,and prefetch
    dataset_type = dataset.get_dataset_type()
    parts, limbs, data_format = train_model.pose_model.parts, train_model.pose_model.limbs, train_model.pose_model.data_format
    flip_list = get_flip_list(dataset_type)
    paramed_map_fn = get_paramed_map_fn(hin, win, hout, wout, parts, limbs, flip_list=flip_list,
                                        data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
    train_dataset = train_dataset.map(paramed_map_fn, num_parallel_calls=max(multiprocessing.cpu_count() // 2, 1))
    train_dataset = train_dataset.batch(config.train.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Load Face Recognition Model
    """
    arcface_ckpt_path = tf.train.latest_checkpoint('arcface-tf2/checkpoints/' + "arc_res50")
    arcface_cfg = load_yaml("arcface_model/config_arc_res50.yaml")
    faceRecModel_Inference = ArcFaceModel(size=arcface_cfg['input_size'],
                                          backbone_type=arcface_cfg['backbone_type'],
                                          training=False)
    faceRecModel_Inference.load_weights(arcface_ckpt_path).expect_partial()
    faceRecModel_Inference.trainable = False
    faceRecModel_Training = None
    if config.train.two_face_rec_models:
        print("[Info] The Face Recognition Model will be trained \n")
        faceRecModel_Training = ArcFaceModel(size=arcface_cfg['input_size'],
                                             backbone_type=arcface_cfg['backbone_type'],
                                             embd_shape=arcface_cfg['embd_shape'],
                                             w_decay=arcface_cfg['w_decay'],
                                             training=False)  # To not include the Head

        # To initialize the model in order to use it in Tensorlayer
        _ = faceRecModel_Training(
            np.random.random((1, arcface_cfg['input_size'], arcface_cfg['input_size'], 3)).astype(np.float32))
        faceRecModel_Training.load_weights(arcface_ckpt_path).expect_partial()
        faceRecModel_Training = tl.layers.Lambda(faceRecModel_Training, faceRecModel_Training.trainable_variables,
                                                 name="arcModel_training")
    else:
        print("[Info] The Face Recognition Model will NOT be trained \n")
    """

    # train configure
    step = tf.Variable(1, trainable=False)
    lr = tf.Variable(lr_init, trainable=False)
    lr_init = tf.Variable(lr_init, trainable=False)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    # opt_face_rec = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    # domain adaptation params
    domainadapt_flag = config.data.domainadapt_flag
    if (domainadapt_flag):
        print("domain adaptaion enabled!")
        discriminator = get_discriminator(train_model)
        opt_d = tf.keras.optimizers.Adam(learning_rate=lr)
        lambda_d = tf.Variable(1, trainable=False)
        ckpt = tf.train.Checkpoint(step=step, optimizer=opt, lr=lr, optimizer_d=opt_d, lambda_d=lambda_d)
    else:
        ckpt = tf.train.Checkpoint(step=step, optimizer=opt, lr=lr)
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)

    # load from ckpt
    try:
        log("loading ckpt...")
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")
    # load pretrained backbone
    try:
        log("loading pretrained backbone...")
        tl.files.load_and_assign_npz_dict(name=pretrain_model_path, network=train_model.backbone, skip=True)
    except:
        log("pretrained backbone doesn't exist, model backbone are initialized")
    # load model weights
    try:
        log("loading saved training model weights...")
        train_model.load_weights(os.path.join(model_dir, "newest_model.npz"))
    except:
        log("model_path doesn't exist, model parameters are initialized")
    if (domainadapt_flag):
        try:
            log("loading saved domain adaptation discriminator weight...")
            discriminator.load_weights(os.path.join(model_dir, "newest_discriminator.npz"))
        except:
            log("discriminator path doesn't exist, discriminator parameters are initialized")

    # uncomment when run from scratch
    for lr_decay_step in lr_decay_steps:
        if (step > lr_decay_step):
            lr = lr * lr_decay_factor

    # lr = lr * (lr_decay_factor ** 2)  # comment if the above lines are uncommented

    # optimize one step
    def one_step(image, gt_conf, gt_paf, mask, train_model, face_boxes, faceRecModel_Inf, faceRecModel_Train):
        step.assign_add(1)
        globals_vars.TFStep = step  # step.numpy()
        with tf.GradientTape() as tape:
            # with tf.GradientTape() as tape, tf.GradientTape() as face_rec_tape:
            x_sensor, pose_out = train_model.forward(image, is_train=True)
            pd_conf, pd_paf, stage_confs, stage_pafs = pose_out[0], pose_out[1], pose_out[2], pose_out[3]
            # Calculate the losses
            optics_loss = train_model.cal_optics_loss(image, x_sensor)
            """
            pd_loss, loss_confs, loss_pafs = train_model.pose_model.cal_loss(gt_conf, gt_paf, mask, stage_confs,
                                                                             stage_pafs)
                                                                             
            re_loss, re_ops_loss = regulize_loss(train_model, weight_decay_factor)
            total_loss_f_b = 1.0 * (pd_loss + re_loss) - 0.8 * (optics_loss + re_ops_loss)                                                           
            """
            pd_loss_face, pd_loss_body, loss_confs_face, loss_confs_body, loss_pafs_face, loss_pafs_body \
                = train_model.cal_split_loss(gt_conf, gt_paf, mask, stage_confs, stage_pafs)

            def sum_arrays(A, B):
                C = []
                for i in range(len(A)):
                    C.append(A[i] + B[i])
                return C

            loss_confs = sum_arrays(loss_confs_face, loss_confs_body)
            loss_pafs = sum_arrays(loss_pafs_face, loss_pafs_body)
            re_loss = 0
            total_loss_f_b = 1.0 * pd_loss_body - 0.8 * optics_loss  # ori
            # total_loss_f_b = 1.0 * pd_loss_body - 0.5 * optics_loss - 0.05 * pd_loss_face  # Minus Face

            # with face rec loss
            # face_rec_loss = calc_face_recognition_loss(image, x_sensor, face_boxes, faceRecModel_Inf,
            #                                            faceRecModel_Train)
            # total_loss_f_b = 1.0 * pd_loss_body - 0.1 * optics_loss - 80.0 * face_rec_loss  # with optics Loss
            # total_loss_f_b = 2.0 * pd_loss_body - 0.2 * optics_loss - 80.0 * face_rec_loss - 0.1 * pd_loss_face  # Mix
            # total_loss_f_b = 1.0 * pd_loss_body - 200.0 * face_rec_loss  # No optics loss
            # total_loss_f_b = 1.0 * pd_loss_body - 0.2 * optics_loss - 100.0 * face_rec_loss - 0.05 * pd_loss_face  # All
            # total_loss_f_b = 0.8 * (pd_loss_body + pd_loss_face) - 0.4 * optics_loss
            # total_loss_f_b = 0.5 * pd_loss_body + 0.5 * re_loss - 0.2 * pd_loss_face - 0.1 * optics_loss - 0.2 * re_ops_loss

            # Two optimizers loss
            # total_loss_f_b = 1.0 * pd_loss_body - 0.8 * optics_loss
            # total_loss_face_rec = - 100.0 * face_rec_loss

        """
        train_weights = []
        for t_weight in train_model.trainable_weights[:last_tuned_layer]:
            if not t_weight.name.startswith('batchnorm'):
                train_weights.append(t_weight)

        gradients = tape.gradient(total_loss, train_weights)
        opt.apply_gradients(zip(gradients, train_weights))
        """
        # trainable_weights[:last_tuned_layer]
        if faceRecModel_Train is not None:
            train_weights = train_model.trainable_weights[:last_tuned_layer] + faceRecModel_Train.trainable_weights
        else:
            train_weights = train_model.trainable_weights[:last_tuned_layer]

        # train_weights = train_model.trainable_weights[:last_tuned_layer]
        gradients = tape.gradient(total_loss_f_b, train_weights)
        # gradients_face_rec = face_rec_tape.gradient(total_loss_face_rec, faceRecModel_Train.trainable_weights)
        opt.apply_gradients(zip(gradients, train_weights))
        # opt_face_rec.apply_gradients(zip(gradients_face_rec, faceRecModel_Train.trainable_weights))
        return pd_conf, pd_paf, total_loss_f_b, re_loss, loss_confs, loss_pafs, optics_loss, x_sensor

    @tf.function
    def one_step_domainadpat(image, gt_conf, gt_paf, mask, labeled, train_model, discriminator, lambda_d):
        step.assign_add(1)
        with tf.GradientTape(persistent=True) as tape:
            # optimize train model
            pd_conf, pd_paf, stage_confs, stage_pafs, backbone_fatures = train_model.forward(image, is_train=True,
                                                                                             domainadapt=True)
            d_predict = discriminator.forward(backbone_fatures)
            pd_loss, loss_confs, loss_pafs = train_model.cal_loss(gt_conf, gt_paf, mask, stage_confs, stage_pafs)
            re_loss = regulize_loss(train_model, weight_decay_factor)
            gan_loss = lambda_d * tf.nn.sigmoid_cross_entropy_with_logits(logits=d_predict, labels=1 - labeled)
            total_loss = pd_loss + re_loss + gan_loss
            d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_predict, labels=labeled)
        # optimize G
        g_gradients = tape.gradient(total_loss, train_model.trainable_weights)
        opt.apply_gradients(zip(g_gradients, train_model.trainable_weights))
        # optimize D
        d_gradients = tape.gradient(d_loss, discriminator.trainable_weights)
        opt_d.apply_gradients(zip(d_gradients, discriminator.trainable_weights))
        return pd_conf, pd_paf, total_loss, re_loss, loss_confs, loss_pafs, gan_loss, d_loss

    # train each step
    tic = time.time()
    train_model.train()
    conf_losses, paf_losses = np.zeros(shape=(6)), np.zeros(shape=(6))
    avg_conf_loss, avg_paf_loss, avg_total_loss, avg_re_loss = 0, 0, 0, 0
    avg_gan_loss, avg_d_loss = 0, 0
    log(
        'Start - n_step: {} batch_size: {} lr_init: {} lr_decay_steps: {} lr_decay_factor: {} weight_decay_factor: {}'.format(
            n_step, batch_size, lr_init.numpy(), lr_decay_steps, lr_decay_factor, weight_decay_factor))
    for image, gt_label, mask, labeled, face_boxes in train_dataset:
        # extract gt_label
        if (train_model.pose_model.data_format == "channels_first"):
            gt_conf = gt_label[:, :n_pos, :, :]
            gt_paf = gt_label[:, n_pos:, :, :]
        else:
            gt_conf = gt_label[:, :, :, :n_pos]
            gt_paf = gt_label[:, :, :, n_pos:]
        # learning rate decay
        if (step in lr_decay_steps):
            new_lr_decay = lr_decay_factor ** (lr_decay_steps.index(step) + 1)
            lr = lr_init * new_lr_decay

        # optimize one step
        if (domainadapt_flag):
            lambda_d = 2 / (1 + tf.math.exp(-10 * (step / n_step))) - 1
            pd_conf, pd_paf, total_loss, re_loss, loss_confs, loss_pafs, gan_loss, d_loss = \
                one_step_domainadpat(image.numpy(), gt_conf.numpy(), gt_paf.numpy(), mask.numpy(), labeled.numpy(),
                                     train_model, discriminator, lambda_d)
            avg_gan_loss += gan_loss / log_interval
            avg_d_loss += d_loss / log_interval
        else:
            pd_conf, pd_paf, total_loss, re_loss, loss_confs, loss_pafs, optics_loss, x_sensor = one_step(
                image,
                gt_conf,
                gt_paf,
                mask,
                train_model,
                face_boxes,
                None,
                None)
            # print("==> TIEMPO DE UNA ITERACION: {}".format(time.time() - tic))
            # tic = time.time()

        avg_conf_loss += tf.reduce_mean(loss_confs) / batch_size / log_interval
        avg_paf_loss += tf.reduce_mean(loss_pafs) / batch_size / log_interval
        avg_total_loss += total_loss / log_interval
        avg_re_loss += re_loss / log_interval

        # debug
        """
        for stage_id, (loss_conf, loss_paf) in enumerate(zip(loss_confs, loss_pafs)):
            conf_losses[stage_id] += loss_conf / batch_size / log_interval
            paf_losses[stage_id] += loss_paf / batch_size / log_interval
        """

        # save log info periodly
        if ((step.numpy() != 0) and (step.numpy() % log_interval) == 0):
            log(
                'Train iteration {} / {}: Learning rate {} total_loss:{}, conf_loss:{}, paf_loss:{}, l2_loss {} stage_num:{} time:{}'.format(
                    step.numpy(), n_step, lr.numpy(), avg_total_loss, avg_conf_loss, avg_paf_loss, avg_re_loss,
                    len(loss_confs), time.time() - tic))
            tic = time.time()
            figure = probe_model(train_model, step.numpy(), config)
            experiment.log({
                'loss': avg_total_loss,
                'lr': lr.numpy(),
                'l2_loss': avg_re_loss,
                'test_image_estimation': wandb.Image(figure),
                'Step': step.numpy()
            })
            save_figs(train_model, step.numpy(), config, "test_img1")
            save_figs(train_model, step.numpy(), config, "test_img2")
            save_figs(train_model, step.numpy(), config, "test_img3")

            tf.config.run_functions_eagerly(True)
            '''
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', avg_total_loss, step=step.numpy(), description='Total loss')
                tf.summary.scalar('lr', lr.numpy(), step=step.numpy(), description='Learning rate')
                tf.summary.scalar('l2_loss', avg_re_loss, step=step.numpy(), description='L2 Loss')
                figure = probe_model(train_model, step.numpy(), config)
                tf.summary.image("Test Image Estimation", plot_to_image(figure), step=step.numpy())
                save_figs(train_model, step.numpy(), config, "test_img1")
                save_figs(train_model, step.numpy(), config, "test_img2")
                save_figs(train_model, step.numpy(), config, "test_img3")
            '''
            """ Uncomment for more debugging
            kpts_fig, pafs_fig = plot_body_features(train_model, step.numpy(), config, True)
            if kpts_fig is not None and pafs_fig is not None:
                tf.summary.image("Keypoints Features", kpts_fig, step=step.numpy())
                tf.summary.image("PAFS Features", pafs_fig, step=step.numpy())

            for stage_id in range(0, len(loss_confs)):
                log(f"stage_{stage_id} conf_loss:{conf_losses[stage_id]} paf_loss:{paf_losses[stage_id]}")
            """
            if (domainadapt_flag):
                log(f"adaptation loss: g_loss:{avg_gan_loss} d_loss:{avg_d_loss}")

            avg_total_loss, avg_conf_loss, avg_paf_loss, avg_re_loss = 0, 0, 0, 0
            avg_gan_loss, avg_d_loss = 0, 0
            # conf_losses, paf_losses = np.zeros(shape=(6)), np.zeros(shape=(6)) # for debugging

        if ((step.numpy() != 0) and (step.numpy() % 5000) == 0):
            log("Saving model every 5000 steps")
            model_save_path = os.path.join(model_dir, "partial_results/model_" + str(step.numpy()) + ".npz")
            train_model.save_weights(model_save_path)
            log(f"model save_path:{model_save_path} saved!\n")

        # save result and ckpt periodly
        if ((step.numpy() != 0) and (step.numpy() % save_interval) == 0):
            # save ckpt
            log("saving model ckpt and result...")
            ckpt_save_path = ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            # save train model
            model_save_path = os.path.join(model_dir, "newest_model.npz")
            train_model.save_weights(model_save_path)
            log(f"model save_path:{model_save_path} saved!\n")
            # save discriminator model
            if (domainadapt_flag):
                dis_save_path = os.path.join(model_dir, "newest_discriminator.npz")
                discriminator.save_weights(dis_save_path)
                log(f"discriminator save_path:{dis_save_path} saved!\n")
            # draw result
            try:
                draw_results(x_sensor.numpy(), gt_conf.numpy(), pd_conf.numpy(), gt_paf.numpy(), pd_paf.numpy(),
                             mask.numpy(), vis_dir, 'train_%d_' % step, data_format=data_format)
            except:
                draw_results(x_sensor.numpy(), gt_conf, pd_conf, gt_paf, pd_paf, mask.numpy(), vis_dir,
                             'train_%d_' % step, data_format=data_format)

            figure = probe_model(train_model, step.numpy(), config)
            matplotlib.pyplot.savefig(os.path.join(vis_dir, 'train_test_img_%d.png' % step), dpi=300)

        # training finished
        if (step == n_step):
            break


def parallel_train(train_model, dataset, config):
    '''Parallel train pipeline of openpose class models

    input model and dataset, the train pipeline will start automaticly
    the train pipeline will:
    1.store and restore ckpt in directory ./save_dir/model_name/model_dir
    2.log loss information in directory ./save_dir/model_name/log.txt
    3.visualize model output periodly during training in directory ./save_dir/model_name/train_vis_dir
    the newest model is at path ./save_dir/model_name/model_dir/newest_model.npz

    Parameters
    ----------
    arg1 : tensorlayer.models.MODEL
        a preset or user defined model object, obtained by Model.get_model() function
    
    arg2 : dataset
        a constructed dataset object, obtained by Dataset.get_dataset() function
    
    
    Returns
    -------
    None
    '''

    init_log(config)
    # train hyper params
    # dataset params
    n_step = config.train.n_step
    batch_size = config.train.batch_size
    # learning rate params
    lr_init = config.train.lr_init
    lr_decay_factor = config.train.lr_decay_factor
    lr_decay_steps = [200000, 300000, 360000, 420000, 480000, 540000, 600000, 700000, 800000, 900000]
    weight_decay_factor = config.train.weight_decay_factor
    # log and checkpoint params
    log_interval = config.log.log_interval
    save_interval = config.train.save_interval
    vis_dir = config.train.vis_dir

    # model hyper params
    n_pos = train_model.n_pos
    hin = train_model.hin
    win = train_model.win
    hout = train_model.hout
    wout = train_model.wout
    model_dir = config.model.model_dir
    pretrain_model_dir = config.pretrain.pretrain_model_dir
    pretrain_model_path = f"{pretrain_model_dir}/newest_{train_model.backbone.name}.npz"

    # import kungfu
    from kungfu import current_cluster_size, current_rank
    from kungfu.tensorflow.initializer import broadcast_variables
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer, SynchronousAveragingOptimizer, \
        PairAveragingOptimizer

    print(f"parallel training using learning rate:{lr_init} batch_size:{batch_size}")
    # training dataset configure with shuffle,augmentation,and prefetch
    train_dataset = dataset.get_train_dataset()
    dataset_type = dataset.get_dataset_type()
    parts, limbs, data_format = train_model.parts, train_model.limbs, train_model.data_format
    flip_list = get_flip_list(dataset_type)
    paramed_map_fn = get_paramed_map_fn(hin, win, hout, wout, parts, limbs, flip_list=flip_list,
                                        data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096)
    train_dataset = train_dataset.shard(num_shards=current_cluster_size(), index=current_rank())
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.map(paramed_map_fn, num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(64)

    # train model configure
    step = tf.Variable(1, trainable=False)
    lr = tf.Variable(lr_init, trainable=False)
    if (config.model.model_type == MODEL.Openpose):
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt = tf.train.Checkpoint(step=step, optimizer=opt, lr=lr)
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)

    # load from ckpt
    try:
        log("loading ckpt...")
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")
    # load pretrained backbone
    try:
        log("loading pretrained backbone...")
        tl.files.load_and_assign_npz_dict(name=pretrain_model_path, network=train_model.backbone, skip=True)
    except:
        log("pretrained backbone doesn't exist, model backbone are initialized")
    # load model weights
    try:
        train_model.load_weights(os.path.join(model_dir, "newest_model.npz"))
    except:
        log("model_path doesn't exist, model parameters are initialized")

    # KungFu configure
    kungfu_option = config.train.kungfu_option
    if kungfu_option == KUNGFU.Sync_sgd:
        print("using Kungfu.SynchronousSGDOptimizer!")
        opt = SynchronousSGDOptimizer(opt)
    elif kungfu_option == KUNGFU.Sync_avg:
        print("using Kungfu.SynchronousAveragingOptimize!")
        opt = SynchronousAveragingOptimizer(opt)
    elif kungfu_option == KUNGFU.Pair_avg:
        print("using Kungfu.PairAveragingOptimizer!")
        opt = PairAveragingOptimizer(opt)

    n_step = n_step // current_cluster_size() + 1  # KungFu
    for step_idx, step in enumerate(lr_decay_steps):
        lr_decay_steps[step_idx] = step // current_cluster_size() + 1  # KungFu

    # optimize one step
    @tf.function
    def one_step(image, gt_label, mask, train_model, is_first_batch=False):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            gt_conf = gt_label[:, :n_pos, :, :]
            gt_paf = gt_label[:, n_pos:, :, :]
            pd_conf, pd_paf, stage_confs, stage_pafs = train_model.forward(image, is_train=True)

            pd_loss, loss_confs, loss_pafs = train_model.cal_loss(gt_conf, gt_paf, mask, stage_confs, stage_pafs)
            re_loss = regulize_loss(train_model, weight_decay_factor)
            total_loss = pd_loss + re_loss

        gradients = tape.gradient(total_loss, train_model.trainable_weights)
        opt.apply_gradients(zip(gradients, train_model.trainable_weights))
        # Kung fu
        if (is_first_batch):
            broadcast_variables(train_model.all_weights)
            broadcast_variables(opt.variables())
        return gt_conf, gt_paf, pd_conf, pd_paf, total_loss, re_loss

    # train each step
    tic = time.time()
    train_model.train()
    log(f"Worker {current_rank()}: Initialized")
    log('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_steps: {} lr_decay_factor: {}'.format(
        n_step, batch_size, lr_init, lr_decay_steps, lr_decay_factor))
    for image, gt_label, mask in train_dataset:
        # learning rate decay
        if (step in lr_decay_steps):
            new_lr_decay = lr_decay_factor ** (float(lr_decay_steps.index(step) + 1))
            lr = lr_init * new_lr_decay
        # optimize one step
        gt_conf, gt_paf, pd_conf, pd_paf, total_loss, re_loss = one_step(image.numpy(), gt_label.numpy(), mask.numpy(), \
                                                                         train_model, step == 0)
        # save log info periodly
        if ((step.numpy() != 0) and (step.numpy() % log_interval) == 0):
            tic = time.time()
            log('Total Loss at iteration {} / {} is: {} Learning rate {} l2_loss {} time:{}'.format(
                step.numpy(), n_step, total_loss, lr.numpy(), re_loss, time.time() - tic))

        # save result and ckpt periodly
        if ((step != 0) and (step % save_interval) == 0 and current_rank() == 0):
            log("saving model ckpt and result...")
            draw_results(image.numpy(), gt_conf.numpy(), pd_conf.numpy(), gt_paf.numpy(), pd_paf.numpy(), mask.numpy(), \
                         vis_dir, 'train_%d_' % step)
            ckpt_save_path = ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            model_save_path = os.path.join(model_dir, "newest_model.npz")
            train_model.save_weights(model_save_path)
            log(f"model save_path:{model_save_path} saved!\n")

        # training finished
        if (step == n_step):
            break
