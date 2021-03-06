import os
import cv2
import json
import numpy as np
import tensorflow as tf
import scipy.stats as st
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
from .infer import Post_Processor
from .utils import draw_results
from ..common import pad_image
from train_utils import psnr
from skimage.metrics import structural_similarity as compare_ssim

def multiscale_search(img, model, private_net=False):
    scales = [0.5, 1.0, 1.5, 2.0]
    img_h, img_w, _ = img.shape
    if private_net:
        pose_model = model.pose_model
    else:
        pose_model = model
    hin, win = pose_model.hin, pose_model.win
    stride = pose_model.hin / pose_model.hout
    data_format = pose_model.data_format
    avg_conf_map = 0
    avg_paf_map = 0
    for scale in scales:
        # currently we scale image without tortion and use padding to make the image_size can be divided by stride
        scale_rate = min(scale * pose_model.hin / img_h, scale * pose_model.win / img_w)
        scale_h, scale_w = int(scale_rate * img_h), int(scale_rate * img_w)
        scaled_img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_CUBIC)
        padded_img, pad = pad_image(scaled_img, stride, pad_value=0.0)
        padded_h, padded_w, _ = padded_img.shape
        input_img = padded_img[np.newaxis, :, :, :].astype(np.float32)
        if (data_format == "channels_first"):
            input_img = input_img.transpose([0, 3, 1, 2])
        input_img = tf.convert_to_tensor(input_img)

        if private_net:
            x_sensor, pose_out = model.infer(input_img)
            conf_map, paf_map = pose_out[0], pose_out[1]
        else:
            x_sensor = input_img
            conf_map, paf_map = model.infer(input_img)

        conf_map = conf_map.numpy()[0]
        paf_map = paf_map.numpy()[0]
        if (data_format == "channels_first"):
            conf_map = np.transpose(conf_map, [1, 2, 0])
            paf_map = np.transpose(paf_map, [1, 2, 0])
        # conf_map restore
        conf_map = cv2.resize(conf_map, (padded_w, padded_h), interpolation=cv2.INTER_CUBIC)
        conf_map = conf_map[pad[0]:pad[0] + scale_h, pad[2]:pad[2] + scale_w, :]
        conf_map = cv2.resize(conf_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        # paf_map restore
        paf_map = cv2.resize(paf_map, (padded_w, padded_h), interpolation=cv2.INTER_CUBIC)
        paf_map = paf_map[pad[0]:pad[0] + scale_h, pad[2]:pad[2] + scale_w, :]
        paf_map = cv2.resize(paf_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        if (data_format == "channels_first"):
            conf_map = np.transpose(conf_map, [2, 0, 1])
            paf_map = np.transpose(paf_map, [2, 0, 1])
        # average
        avg_conf_map += conf_map / (len(scales))
        avg_paf_map += paf_map / (len(scales))
    # tf.print(f"test shape in multi: conf_map:{avg_conf_map.shape} paf_map:{avg_paf_map.shape}")
    return x_sensor, avg_conf_map, avg_paf_map


def infer_one_img(model, post_processor, img, img_id=-1, enable_multiscale_search=False, is_visual=False,
                  save_dir="./vis_dir"):
    img = img.numpy().astype(np.float32)
    img_id = img_id.numpy()
    img_h, img_w, _ = img.shape
    private_net = False
    if hasattr(model, 'pose_model'):
        private_net = True
        pose_model = model.pose_model
    else:
        pose_model = model
    data_format = pose_model.data_format
    if (enable_multiscale_search):
        x_sensor, conf_map, paf_map = multiscale_search(img, model, private_net)
    else:
        input_img = cv2.resize(img, (pose_model.win, pose_model.hin))[np.newaxis, :, :, :]
        if (data_format == "channels_first"):
            input_img = input_img.transpose([0, 3, 1, 2])
        if private_net:
            x_sensor, pose_out = model.infer(input_img)
            conf_map, paf_map = pose_out[0], pose_out[1]
        else:
            x_sensor = input_img
            conf_map, paf_map = model.infer(input_img)
        conf_map = conf_map.numpy()[0]
        paf_map = paf_map.numpy()[0]
    x_sensor = tf.image.resize(x_sensor, (img_h, img_w))[0].numpy()
    humans = post_processor.process(conf_map.copy(), paf_map.copy(), img_h, img_w, data_format=data_format)
    # tf.print(f"test shape before draw: img:{img.shape} conf_map:{conf_map.shape} paf_map:{paf_map.shape}")
    if (is_visual):
        if (data_format == "channels_first"):
            conf_map = conf_map.transpose([1, 2, 0])
            paf_map = paf_map.transpose([1, 2, 0])
        draw_conf_map = cv2.resize(conf_map, (img_w, img_h))
        draw_paf_map = cv2.resize(paf_map, (img_w, img_h))
        visualize(x_sensor, img_id, humans, draw_conf_map, draw_paf_map, save_dir)
    psnr_res = psnr(img, x_sensor)
    ssim_res = compare_ssim(img, x_sensor, multichannel=True)
    return humans, psnr_res, ssim_res


def visualize(img, img_id, humans, conf_map, paf_map, save_dir):
    tf.print(f"{len(humans)} human found!")
    tf.print("visualizing...")
    os.makedirs(save_dir, exist_ok=True)
    ori_img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    vis_img = ori_img.copy()
    for human in humans:
        vis_img = human.draw_human(vis_img)
    fig = plt.figure(figsize=(8, 8))
    # show input image
    a = fig.add_subplot(2, 2, 1)
    a.set_title("input image")
    plt.imshow(ori_img)
    # show output result
    a = fig.add_subplot(2, 2, 2)
    a.set_title("output result")
    plt.imshow(vis_img)
    # show conf_map
    show_conf_map = np.amax(conf_map[:, :, :-1], axis=2)
    a = fig.add_subplot(2, 2, 3)
    a.set_title("conf_map")
    plt.imshow(show_conf_map, alpha=0.8)
    plt.colorbar()
    # show paf_map
    show_paf_map = np.amax(paf_map[:, :, :], axis=2)
    a = fig.add_subplot(2, 2, 4)
    a.set_title("paf_map")
    plt.imshow(show_paf_map, alpha=0.8)
    plt.colorbar()
    # save
    plt.savefig(f"{save_dir}/{img_id}_visualize.png")
    plt.close('all')


def _map_fn(image_file, image_id, hin, win):
    # load data
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, image_id


def evaluate(model, dataset, config, vis_num=30, total_eval_num=30, enable_multiscale_search=True):
    '''evaluate pipeline of Openpose class models

    input model and dataset, the evaluate pipeline will start automaticly
    the evaluate pipeline will:
    1.loading newest model at path ./save_dir/model_name/model_dir/newest_model.npz
    2.perform inference and parsing over the chosen evaluate dataset
    3.visualize model output in evaluation in directory ./save_dir/model_name/eval_vis_dir
    4.output model metrics by calling dataset.official_eval()

    Parameters
    ----------
    arg1 : tensorlayer.models.MODEL
        a preset or user defined model object, obtained by Model.get_model() function

    arg2 : dataset
        a constructed dataset object, obtained by Dataset.get_dataset() function

    arg3 : Int
        an Integer indicates how many model output should be visualized

    arg4 : Int
        an Integer indicates how many images should be evaluated

    Returns
    -------
    None
    '''
    tf.print(f"enable multiscale_search:{enable_multiscale_search}")
    if config.model.eval_checkpoint == "newest":
        model.load_weights(os.path.join(config.model.model_dir, "newest_model.npz"))
    else:
        model.load_weights(os.path.join(config.model.model_dir,
                                        "partial_results/model_" + config.model.eval_checkpoint + ".npz"))
    model.eval()
    pd_anns = []
    mean_psnr = []
    mean_ssim = []
    vis_dir = config.eval.vis_dir
    kpt_converter = dataset.get_output_kpt_cvter()
    post_processor = Post_Processor(parts=model.pose_model.parts, limbs=model.pose_model.limbs,
                                    colors=model.pose_model.colors)

    eval_dataset = dataset.get_eval_dataset()
    paramed_map_fn = partial(_map_fn, hin=model.pose_model.hin, win=model.pose_model.win)
    eval_dataset = eval_dataset.map(paramed_map_fn, num_parallel_calls=max(multiprocessing.cpu_count() // 2, 1))
    for eval_num, (img, img_id) in enumerate(eval_dataset):
        if (eval_num >= total_eval_num):
            break
        if (eval_num <= vis_num):
            humans, psnrv, ssimv = infer_one_img(model, post_processor, img, img_id=img_id, is_visual=True, save_dir=vis_dir,
                                   enable_multiscale_search=enable_multiscale_search)
        else:
            humans, psnrv, ssimv = infer_one_img(model, post_processor, img, img_id=img_id, is_visual=False, save_dir=vis_dir,
                                   enable_multiscale_search=enable_multiscale_search)
        mean_psnr.append(psnrv)
        mean_ssim.append(ssimv)
        for human in humans:
            ann = {}
            ann["category_id"] = 1
            ann["image_id"] = int(img_id.numpy())
            ann["id"] = human.get_global_id()
            ann["area"] = human.get_area()
            ann["score"] = human.get_score()
            kpt_list = []
            for part_idx in range(0, model.pose_model.n_pos):
                if (part_idx not in human.body_parts) or (part_idx in [14, 15, 16, 17]):
                    kpt_list.append([-1000, -1000])
                else:
                    body_part = human.body_parts[part_idx]
                    kpt_list.append([body_part.get_x(), body_part.get_y()])
            ann["keypoints"] = kpt_converter(kpt_list)
            pd_anns.append(ann)
        if (eval_num % 100 == 0):
            tf.print(f"evaluating {eval_num}/{len(list(eval_dataset))}")

    print("Mean PSNR: {}, MEAN SSIM: {}".format(np.array(mean_psnr).mean(), np.array(mean_ssim).mean()))
    result_dic = {"annotations": pd_anns}
    dataset.official_eval(result_dic, vis_dir)
