import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import globals as globals_var
import math
import datetime
import cv2
import os
from pathlib import Path
# from deepface.commons.functions import detect_face
# from deepface import DeepFace
from hyperpose.Model.openpose.infer import Post_Processor

post_processor = None
test_img1 = None
test_img2 = None
test_img3 = None
img_height = None
img_width = None


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def now():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # strftime("%d%a%m%y-%H%M")


def get_test_img_data(img_path, hin, win):
    img = cv2.imread(img_path)  # B,G,R order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to Tensor
    input_img = tf.image.resize(img, (hin, win))
    input_img = tf.image.convert_image_dtype(input_img, tf.float32)
    input_img /= 255  # normalize image [0,1]
    input_img = input_img[tf.newaxis, ...]
    # input_img = tf.transpose(input_img, [0, 3, 1, 2])
    return input_img


def visualize(img, humans, conf_map, paf_map, save_dir=None, save_name="tmp_fig"):
    print(f"{len(humans)} human found!")
    print("visualizing...")
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
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
    show_conf_map = np.amax(np.abs(conf_map[:-1, :, :]), axis=0)
    a = fig.add_subplot(2, 2, 3)
    a.set_title("conf_map")
    plt.imshow(show_conf_map)
    # show paf_map
    show_paf_map = np.amax(np.abs(paf_map[:, :, :]), axis=0)
    a = fig.add_subplot(2, 2, 4)
    a.set_title("paf_map")
    plt.imshow(show_paf_map)
    # save
    if save_dir is not None:
        plt.savefig(f"{save_dir}/{save_name}_visualize.png")
        plt.close('all')
    else:
        plt.show()


def set_probe_params(model, cfg):
    global post_processor, test_img1, test_img2, test_img3, img_height, img_width
    post_processor = Post_Processor(parts=model.pose_model.parts, limbs=model.pose_model.limbs,
                                    colors=model.pose_model.colors)
    img_height, img_width = model.pose_model.hin, model.pose_model.win
    test_img1 = get_test_img_data(cfg.log.test_img1, img_height, img_width)
    test_img2 = get_test_img_data(cfg.log.test_img2, img_height, img_width)
    test_img3 = get_test_img_data(cfg.log.test_img3, img_height, img_width)


def save_figs(model, step, cfg, img_name):
    # Stop Summaries from utils.py
    globals_var.stopUtilsSummary = True
    # load global vars
    global post_processor, test_img1, test_img2, test_img3, img_height, img_width

    if img_name == "test_img1":
        # Net Output
        y_sensor, pose_out = model.infer(test_img1)
        conf_map = pose_out[0].numpy()
        paf_map = pose_out[1].numpy()

        humans = post_processor.process(conf_map[0], paf_map[0], img_height, img_width,
                                        data_format=model.pose_model.data_format)
        input_img = test_img1  # tf.transpose(test_img1, [0, 2, 3, 1])

    if img_name == "test_img2":
        # Net Output
        y_sensor, pose_out = model.infer(test_img2)
        conf_map = pose_out[0].numpy()
        paf_map = pose_out[1].numpy()

        humans = post_processor.process(conf_map[0], paf_map[0], img_height, img_width,
                                        data_format=model.pose_model.data_format)
        input_img = test_img2

    if img_name == "test_img3":
        # Net Output
        y_sensor, pose_out = model.infer(test_img3)
        conf_map = pose_out[0].numpy()
        paf_map = pose_out[1].numpy()

        humans = post_processor.process(conf_map[0], paf_map[0], img_height, img_width,
                                        data_format=model.pose_model.data_format)
        input_img = test_img3


    # Normalize Results for plotting
    y_sensor = y_sensor[0].numpy()
    input_img = input_img.numpy()[0]
    if y_sensor.min() >= 0:
        y_sensor = cv2.normalize(y_sensor, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_32F)
    else:
        print("[WARNING] Sensor image has negative values")
        old_min = y_sensor.min()
        old_range = y_sensor.max() - old_min
        new_min = 0
        new_range = 1 - new_min
        y_sensor = (y_sensor - old_min) / old_range * new_range + new_min
        # print("after normalize Sensor image is [{},{}]".format(y_sensor.min(), y_sensor.max()))

    # Plot Outputs
    figure, ax = plt.subplots(1, 2)
    figure.set_figheight(6)
    figure.set_figwidth(10)
    # figure.suptitle('Test Results Step %d' % (step + 1), fontsize=14)

    # Sensor Image
    ax[0].set_title('Sensor Image \n PSNR: %.4f' % psnr(input_img, y_sensor))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].grid(False)
    ax[0].set_axis_off()
    ax[0].imshow(y_sensor)

    # Skeleton Image
    skel_img = y_sensor.copy()
    for human in humans:
        skel_img = human.draw_human(skel_img)

    ax[1].set_title('Skeleton Image \n Body parts detected: %d' % len(humans))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].grid(False)
    ax[1].set_axis_off()
    ax[1].imshow(skel_img)

    plt.savefig(globals_var.save_path + img_name + "_" + str(step) + ".png", dpi=300)

    # Allow Summaries from utils.py
    globals_var.stopUtilsSummary = False


def probe_model(model, step, cfg):
    # create temp folder
    tmp_folder = "/tmp/"+now()
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    # Stop Summaries from utils.py
    globals_var.stopUtilsSummary = True
    # load global vars
    global post_processor, test_img1, test_img2, test_img3, img_height, img_width
    # Net Output
    y_sensor, pose_out = model.infer(test_img1)
    conf_map = pose_out[0].numpy()
    paf_map = pose_out[1].numpy()

    humans = post_processor.process(conf_map[0], paf_map[0], img_height, img_width,
                                    data_format=model.pose_model.data_format)
    draw_conf_map = cv2.resize(conf_map[0].transpose([1, 2, 0]), (img_width, img_height)).transpose([2, 0, 1])
    draw_paf_map = cv2.resize(paf_map[0].transpose([1, 2, 0]), (img_width, img_height)).transpose([2, 0, 1])

    # y_sensor = tf.transpose(y_sensor, [0, 2, 3, 1])
    input_img = test_img1   # tf.transpose(test_img1, [0, 2, 3, 1])
    # visualize(y_sensor[0], humans, draw_conf_map, draw_paf_map)
    # Normalize Results for plotting
    y_sensor = y_sensor[0].numpy()
    input_img = input_img.numpy()[0]
    if y_sensor.min() >= 0:
        y_sensor = cv2.normalize(y_sensor, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_32F)
    else:
        print("[WARNING] Sensor image has negative values")
        old_min = y_sensor.min()
        old_range = y_sensor.max() - old_min
        new_min = 0
        new_range = 1 - new_min
        y_sensor = (y_sensor - old_min) / old_range * new_range + new_min
        # print("after normalize Sensor image is [{},{}]".format(y_sensor.min(), y_sensor.max()))

    # Plot Outputs
    figure, ax = plt.subplots(2, 2)
    figure.set_figheight(6)
    figure.set_figwidth(10)
    # figure.suptitle('Test Results Step %d' % (step + 1), fontsize=14)

    # Sensor Image
    ax[0][0].set_title('Sensor Image \n PSNR: %.4f' % psnr(input_img, y_sensor))
    ax[0][0].set_xticks([])
    ax[0][0].set_yticks([])
    ax[0][0].grid(False)
    ax[0][0].set_axis_off()
    ax[0][0].imshow(y_sensor)

    # Skeleton Image
    skel_img = y_sensor.copy()
    for human in humans:
        skel_img = human.draw_human(skel_img)

    ax[0][1].set_title('Skeleton Image \n Body parts detected: %d' % len(humans))
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])
    ax[0][1].grid(False)
    ax[0][1].set_axis_off()
    ax[0][1].imshow(skel_img)

    # Face Recognition
    yoff = int((img_height - 224) / 2)
    xoff = int((img_width - 224) / 2)

    input_img_face = input_img.copy()[25:80, 205:255, :]
    input_img_face = cv2.resize(input_img_face, (224, 224), interpolation=cv2.INTER_AREA)
    tmp_input_file = tmp_folder + '/tmp_input.png'
    plt.imsave(tmp_input_file, input_img_face)

    sensor_img_face = y_sensor.copy()[25:80, 205:255, :]
    sensor_img_face = cv2.resize(sensor_img_face, (224, 224), interpolation=cv2.INTER_AREA)
    tmp_sensor_file = tmp_folder + '/tmp_sensor.png'
    plt.imsave(tmp_sensor_file, sensor_img_face)

    input_img_face_bg = np.ones((img_height, img_width, 3))
    sensor_img_face_bg = np.ones((img_height, img_width, 3))

    input_img_face_bg[yoff:yoff + 224, xoff:xoff + 224] = input_img_face
    sensor_img_face_bg[yoff:yoff + 224, xoff:xoff + 224] = sensor_img_face

    # Get Face Results
    flag_sensor_face = False
    try:
        detect_face(tmp_folder + '/tmp_sensor.png')
        flag_sensor_face = True
    except Exception as e:
        flag_sensor_face = False

    if flag_sensor_face:
        face_result_sensor = {'verified': False, 'distance': np.inf}
        # face_result_sensor = DeepFace.verify(tmp_sensor_file, tmp_input_file)
    else:
        face_result_sensor = {'verified': False, 'distance': np.inf}

    # Plot Face Results
    ax[1][0].set_title('Sensor Face Image')
    ax[1][0].imshow(sensor_img_face_bg)
    ax[1][0].set_xticks([])
    ax[1][0].set_yticks([])
    ax[1][0].grid(False)
    ax[1][0].set_axis_off()

    ax[1][1].set_title('Original Face Image')
    ax[1][1].imshow(input_img_face_bg)
    ax[1][1].set_xticks([])
    ax[1][1].set_yticks([])
    ax[1][1].grid(False)
    ax[1][1].set_axis_off()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    textstr_sensor = '\n'.join((
        r'Face Detected: %s' % (str(flag_sensor_face),),
        r'Verified: %s' % (str(face_result_sensor['verified']),),
        r'Distance: %.4f' % (face_result_sensor['distance'],)))

    ax[1][0].text(0.19, 0.1, textstr_sensor, transform=ax[1][0].transAxes, fontsize=10, verticalalignment='top',
                  bbox=props)

    """
    # Save Results
    if step % 1999:
        Path(cfg.OUTPUT_DATA_PATH).mkdir(parents=True, exist_ok=True)
        np.savez(cfg.OUTPUT_DATA_PATH + '/result_epoch=%d' % epoch + '_step=%d.npz' % step,
                 y_sensor=y_sensor, skel_img=skel_img, figure=figure, skeletonizer=skeletonizer,
                 kpts=kpts, pafs=pafs, face_result_sensor=face_result_sensor,
                 input_img=input_img)
    """

    # Allow Summaries from utils.py
    globals_var.stopUtilsSummary = False

    return figure


def plot_body_features(model, step, cfg, save_plot=False):
    # Stop Summaries from utils.py
    globals_var.stopUtilsSummary = True
    # load global vars
    global post_processor, test_img1, test_img2, test_img3, img_height, img_width
    # Net Output
    y_sensor, pose_out = model.infer(test_img1)
    kpts = pose_out[0]
    # kpts = tf.transpose(kpts, [0, 2, 3, 1])[0].numpy()
    pafs = pose_out[1]
    # pafs = tf.transpose(pafs, [0, 2, 3, 1])[0].numpy()

    # Plot Outputs
    figure, ax = plt.subplots(3, 6)
    figure.set_figheight(3)
    figure.set_figwidth(6)
    kpts_face_idxs = [0, 14, 15, 16, 17]
    pafs_face_idxs = [15 * 2, 15 * 2 + 1, 16 * 2, 16 * 2 + 1, 17 * 2, 17 * 2 + 1, 18 * 2, 18 * 2 + 1]
    # np.arange(28, 38).tolist()

    # plot kpts
    for i in range(0, 3):
        for j in range(0, 6):
            k = i*6+j
            ax[i][j].set_title("id: {}".format(k), color='r' if k in kpts_face_idxs else 'b')
            ax[i][j].set_xticks([])
            ax[i][j].grid(False)
            ax[i][j].set_axis_off()
            ax[i][j].imshow(kpts[0, ..., k])

    if save_plot:
        plt.savefig(globals_var.save_path + "kpts" + "_" + str(step) + ".png", dpi=300)
        kpts_fig = None
    else:
        kpts_fig = plot_to_image(figure)

    figure2, ax2 = plt.subplots(4, 10)
    figure2.set_figheight(4)
    figure2.set_figwidth(10)

    for i in range(0, 4):
        for j in range(0, 10):
            k = i * 10 + j
            if k > 37:
                ax2[i][j].set_xticks([])
                ax2[i][j].grid(False)
                ax2[i][j].set_axis_off()
            else:
                ax2[i][j].set_title("id: {}".format(k), color='r' if k in pafs_face_idxs else 'b')
                ax2[i][j].set_xticks([])
                ax2[i][j].grid(False)
                ax2[i][j].set_axis_off()
                ax2[i][j].imshow(pafs[0, ..., k])

    if save_plot:
        plt.savefig(globals_var.save_path + "pafs" + "_" + str(step) + ".png", dpi=300)
        pafs_fig = None
    else:
        pafs_fig = plot_to_image(figure2)

    # Allow Summaries from utils.py
    globals_var.stopUtilsSummary = False

    return kpts_fig, pafs_fig
