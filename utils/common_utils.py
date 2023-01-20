import numpy as np
import time
import pickle
import os
import torch

from playsound import playsound

from torchvision.models import ResNet50_Weights
from torch.autograd import Variable
from PIL import Image
import cv2

from utils.pose_utils import *
from yolov5.utils.augmentations import letterbox


def intention_inference(pose_2d_dict, im0, model_pos, res_model, intent_model=None, return_features=False, play_audio=False):
    res_w, res_h = im0.shape[1], im0.shape[0]
    processed_2d_pose = process_2dpose_for_3d(pose_2d_dict, norm_pose_2d=False,
                                              normalize_screen=True, res_w=res_w, res_h=res_h)

    prediction_3d = inference_3d(processed_2d_pose, model_pos)

    res_features_cut, res_features_all = extract_resnet_features(
        im0, res_model, bbox=pose_2d_dict['bbox'][-1])

    if intent_model is not None:

        max_len = 651
        if len(prediction_3d.shape) == 3:
            input_3d_pose = [np.pad(prediction_3d, ((
                0, max_len-len(prediction_3d)), (0, 0), (0, 0))), len(prediction_3d)]
        elif len(prediction_3d.shape) == 2:
            input_3d_pose = [np.pad(
                prediction_3d, ((0, max_len-len(prediction_3d)), (0, 0))), len(prediction_3d)]

        device = next(intent_model.parameters()).device

        inputs = [torch.tensor(d).unsqueeze(0).to(device) if type(d) != list else [torch.tensor(e).unsqueeze(
            0).to(device) for e in d] for d in [input_3d_pose, res_features_cut, res_features_all]]
        output = intent_model(*inputs)

        output = torch.softmax(output, dim=1).squeeze(0).cpu().detach().numpy()

        print(f'Positive: {output[1]} Negative: {output[0]}')

        if play_audio:

            is_played = False

            if output[0] > 0.5:
                playsound('../remind.wav', block=False)
                is_played = True

            if return_features:
                return prediction_3d, res_features_cut, res_features_all, output, is_played

        if return_features:
            return prediction_3d, res_features_cut, res_features_all, output

    if return_features:
        return prediction_3d, res_features_cut, res_features_all


def extract_resnet_features(im0, res_model, bbox=None):

    preprocess = ResNet50_Weights.DEFAULT.transforms()
    device = next(res_model.parameters()).device

    img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    t_img = preprocess(img).unsqueeze(0).to(device)

    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        im_cut = im0[y1:y2, x1:x2].copy()
        img_cut = cv2.cvtColor(im_cut, cv2.COLOR_BGR2RGB)
        img_cut = Image.fromarray(img_cut)
        t_cut = preprocess(img_cut).unsqueeze(0).to(device)

        t_img = torch.cat([t_cut, t_img], dim=0)

    hook_feas = []

    def hook_func(model, input, output):
        for fea in output:
            hook_feas.append(fea.flatten().cpu().detach().numpy())

    layer = res_model._modules.get('avgpool')
    handle = layer.register_forward_hook(hook_func)

    with torch.no_grad():
        opt = res_model(t_img)

    assert len(hook_feas) < 3

    return hook_feas


def process_2dpose_for_3d(pose_2d_dict, norm_pose_2d=False, normalize_screen=True, res_w=None, res_h=None):
    """
    Process 2d pose result for 3d inference.

    Args:
        pose_2d_dict ({'keypoints':List[Array], 'bbox':List[Array]})

        norm_pose_2d (bool) if True, normalize 2d pose point to same scale according to bbox.

        normalize_screen (bool) if True, normalize screen to (-1, 1)
    """
    if pose_2d_dict['keypoints'][0].shape[1] == 3:
        pose_2d_dict['keypoints'] = [pose[:, :2]
                                     for pose in pose_2d_dict['keypoints']]

    pose_2d_dict['keypoints'] = np.array(pose_2d_dict['keypoints'])

    if norm_pose_2d:
        bbox_center = np.array([[528, 427]], dtype=np.float32)
        bbox_scale = 400

        def norm_pose_2d_to_same_scale(pose_2d_dict, bbox_center, bbox_scale):
            kpts_norm = []
            for kpt_frame, bbox_frame in zip(pose_2d_dict['keypoints'], pose_2d_dict['bbox']):
                center = np.array([[(bbox_frame[0] + bbox_frame[2]) / 2,
                                    (bbox_frame[1] + bbox_frame[3]) / 2]])
                scale = max(bbox_frame[2] - bbox_frame[0],
                            bbox_frame[3] - bbox_frame[1])
                kpts_norm.append((kpt_frame[:, :2] - center)
                                 / scale * bbox_scale + bbox_center)
            pose_2d_dict['keypoints'] = np.array(kpts_norm)

            return pose_2d_dict

        pose_2d_dict = norm_pose_2d_to_same_scale(
            pose_2d_dict, bbox_center, bbox_scale)

    if normalize_screen:
        if res_w is None or res_h is None:
            pose_2d_dict['keypoints'] = normalize_screen_coordinates(
                pose_2d_dict['keypoints'], w=1000, h=1002)
        else:
            pose_2d_dict['keypoints'] = normalize_screen_coordinates(
                pose_2d_dict['keypoints'], w=res_w, h=res_h)

    return pose_2d_dict['keypoints']


def update_person_tracking_dict(person_tracking_dict, pose_det_results):

    detected_track_ids = []

    for pose_det_result in pose_det_results:
        track_id = pose_det_result['track_id']
        detected_track_ids.append(track_id)
        if track_id in person_tracking_dict.keys():
            person_tracking_dict[track_id]['keypoints'].append(
                pose_det_result['keypoints'])
            person_tracking_dict[track_id]['bbox'].append(
                pose_det_result['bbox'])
        else:
            person_tracking_dict[track_id] = {}
            person_tracking_dict[track_id]['keypoints'] = [
                pose_det_result['keypoints']]
            person_tracking_dict[track_id]['bbox'] = [pose_det_result['bbox']]

    delete_keys = []
    for key in person_tracking_dict.keys():
        if key not in detected_track_ids:
            delete_keys.append(key)

    for key in delete_keys:
        del person_tracking_dict[key]
    return person_tracking_dict


def update_imgs_list(imgs_list, person_tracking_dict, img, max_length=750):

    if len(person_tracking_dict) == 0:
        del imgs_list
        imgs_list_ = []
        too_long_ids = []
    else:
        too_long_ids = [pid for pid, person_dict in person_tracking_dict.items() if len(
            person_dict['bbox']) > max_length]
        tracking_lengths = [len(person_dict['bbox']) for person_dict in person_tracking_dict.values(
        ) if len(person_dict['bbox']) <= max_length]
        if len(tracking_lengths) == 0:
            del imgs_list
            imgs_list_ = []
        else:
            max_tracking_length = max(tracking_lengths)
            if max_tracking_length > 1:
                imgs_list_ = imgs_list[-(max_tracking_length-1):]
                del imgs_list
                imgs_list_.append(img)
            elif max_tracking_length == 1:
                del imgs_list
                imgs_list_ = []
                imgs_list_.append(img)
    return imgs_list_, too_long_ids

# 3d inference function define


def inference_3d(keypoints, model_pos):

    metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}
    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(
        keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(
        [4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    # Receptive field: 243 frames for [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    input_keypoints = keypoints.copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=True,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, model_pos, return_predictions=True)

    return prediction


def get_line_equation(x1, y1, x2, y2):
    A = y1-y2
    B = x2-x1
    C = x1*y2-x2*y1
    return A, B, C


def get_distance_point_line(x, y, A, B, C):
    return abs((A*x+B*y+C)/(A**2+B**2)**0.5)


def is_above_line(x, y, A, B, C):
    return B * (A*x+B*y+C) > 0


def crop_image(im, crop_paras):

    x1, y1, x2, y2, mid_x = crop_paras

    assert x1 is not None

    crop_ratio = 0.65
    y_down_ratio = 0.1
    x_left_ratio = 0.27
    A, B, C = get_line_equation(x1, y1, x2, y2)
    mid_y = -(A*mid_x + C) / B

    y_min = mid_y - (crop_ratio-y_down_ratio)*im.shape[0]
    y_max = mid_y + y_down_ratio*im.shape[0]
    x_min = mid_x - x_left_ratio*im.shape[1]
    x_max = mid_x + (crop_ratio-x_left_ratio)*im.shape[1]

    y_min, y_max, x_min, x_max = [int(v) for v in [y_min, y_max, x_min, x_max]]

    im = im[y_min:y_max, x_min:x_max]

    x1, x2, mid_x = [x-x_min for x in [x1, x2, mid_x]]
    y1 = y1 - y_min
    y2 = y2 - y_min

    return im, (x1, y1, x2, y2, mid_x)


def crop_pad_resize(img0, crop_paras, img_size, stride, auto=True):

    img0, new_crop_paras = crop_image(img0, crop_paras)

    # Padded resize
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img0, img, new_crop_paras


def get_crop_paras_from_img(im):
    # x1, y1, x2, y2, mid_x = [None] * 5
    x1 = 3840 * 0.13020833333333334
    y1 = 2160 * 0.7569444444444444
    x2 = 3840 * 0.5989583333333334
    y2 = 2160 * 0.6805555555555556
    mid_x = 3840 * 0.3359375
    x1, y1, x2, y2, mid_x = [int(v) for v in [x1, y1, x2, y2, mid_x]]
    print('Crop paras initialized.')
    return x1, y1, x2, y2, mid_x


def pick_up_target(person_tracking_dict, crop_paras, tracked_ids=[]):
    tracked_ids = [i for i in tracked_ids if i in person_tracking_dict.keys()]
    x1, y1, x2, y2, mid_x = crop_paras
    A, B, C = get_line_equation(x1, y1, x2, y2)
    for track_id, singleperson_track_dict in person_tracking_dict.items():
        if track_id in tracked_ids:
            continue
        bbox_start = singleperson_track_dict['bbox'][0]
        start_above_line = is_above_line(
            (bbox_start[0]+bbox_start[2])/2, bbox_start[3], A, B, C)
        if not start_above_line:
            bbox_end = singleperson_track_dict['bbox'][-1]
            end_above_line = is_above_line(
                (bbox_end[0]+bbox_end[2])/2, bbox_end[3], A, B, C)
            if end_above_line:
                tracked_ids.append(track_id)
                if len(singleperson_track_dict['bbox']) > 20 and len(singleperson_track_dict['bbox']) < 600:
                    return singleperson_track_dict.copy(), track_id, tracked_ids
    return None, None, tracked_ids


def save_result_func(tracking_id, saving_dir, save_path, save_images=None, fps=29.97, save_in_one_file=False):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        assert os.path.isdir(save_path)

    if save_images is not None and save_in_one_file is True:
        file_path = os.path.join(save_path, str(tracking_id)+'_all.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump([saving_dir, save_images], f)

        print(f"Data and video of person #{tracking_id} has been saved.")
    else:
        file_path = os.path.join(save_path, str(tracking_id)+'.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(saving_dir, f)

        print(f"Data of person #{tracking_id} has been saved.")

        if save_images is not None:
            for frame_idx, img in enumerate(save_images):
                if frame_idx == 0:
                    w, h = img.shape[1], img.shape[0]
                    video_file_path = os.path.join(
                        save_path, str(tracking_id)+'.mp4')
                    vid_writer = cv2.VideoWriter(
                        video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(img)
            vid_writer.release()
            print(f"Video of person #{tracking_id} has been saved.")
