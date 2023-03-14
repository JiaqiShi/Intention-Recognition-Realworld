import sys

MMPOSE_PATH = '../mmpose'
YOLO_PATH = '../yolov5'

# mmpose path
sys.path.append(MMPOSE_PATH)
# yolov5 path
sys.path.append(YOLO_PATH)

import os
import pickle
import queue
import time
import warnings
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torchvision.models import ResNet50_Weights, resnet50
from pathlib import Path

from utils.common_utils import *
from model.intention_models import *

# mmpose import
from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model)
from mmpose.datasets import DatasetInfo

# yolov5 import
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadWebcam
from yolov5.utils.general import (LOGGER, check_file, check_img_size,
                                  non_max_suppression, scale_coords)
from yolov5.utils.plots import Annotator, colors

MODEL = {'gcn+mlp+mlp':GCN_MLP_MLP}

class ThreadingLoadWebcam:
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size
        self.stopped = False

        self.q = Queue(maxsize=30)

        self.start()
        print('Start video capture process.')

    def start(self):
        thread = Thread(target=self.update, daemon=True)
        thread.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                break
            if not self.q.full():
                
                ret_val, img0 = self.cap.read()

                if not ret_val:
                    break

                self.q.put((ret_val, img0))

    def release(self):
        self.stopped = True
        self.cap.release()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.q.get()
        # img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0

def get_data_from_dataset(source, webcam, imgsz, stride, cvt_color, pt, Q_dataset, device, half=False, use_crop=False, crop_paras=None, skip=False, start_frame=0):

    if webcam:
        # dataset = ThreadingLoadWebcam(source, img_size=imgsz, stride=stride)
        dataset = LoadWebcam(source, img_size=imgsz, stride=stride, cvt_color=cvt_color)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, cvt_color=cvt_color)

    if hasattr(dataset, 'frames'):
        total_frame_num = dataset.frames
        print('Totoal frame num:', total_frame_num)
    else:
        total_frame_num = None
    percent = 0

    if skip:
        dataset.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    video_fps = dataset.cap.get(cv2.CAP_PROP_FPS)

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        if skip:
            frame_idx = frame_idx + start_frame
        if total_frame_num is not None:
            if int(frame_idx/total_frame_num*100) != percent:
                percent = int(frame_idx/total_frame_num*100)
                print(f'{percent} % processed, {frame_idx}/{total_frame_num}')
        if frame_idx == 0:
            if crop_paras is not None:
                assert len(crop_paras) == 5
                if crop_paras[2] < 1:
                    height_im0, width_im0 = im0s.shape[:2]
                    crop_paras = [int(v*width_im0) if num%2==0 else int(v*height_im0) for num, v in enumerate(crop_paras)]
                else:
                    crop_paras = [int(v) for v in crop_paras]
            else:
                crop_paras = get_crop_paras_from_img(im0s)
        if use_crop:
            im0s, im, new_crop_paras = crop_pad_resize(im0s, crop_paras, imgsz, stride=stride, auto=pt)
        else:
            new_crop_paras = crop_paras

        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        Q_dataset.put((frame_idx, (path, im, im0s, s, new_crop_paras, video_fps)))
    print('All data loaded.')
    while Q_dataset.full():
        time.sleep(1)

def person_detection_yolo(model, Q_dataset, Q_person_det, augment, visualize, 
                        conf_thres, iou_thres, classes, agnostic_nms, max_det, bbox_size_thres=None):

    while True:
        try:
            frame_idx, (path, im, im0s, s, crop_paras, video_fps) = Q_dataset.get(timeout=10)

            # Yolo Inference
            pred = model(im, augment=augment, visualize=visualize)

            # Apply NMS
            # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process Yolo predictions
            person_det_results = []
            for i, det in enumerate(pred):  # per image
                p, im0 = path, im0s.copy()

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape)
                for bbox in det:
                    if bbox_size_thres is not None:
                        bbox_size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                        if bbox_size < bbox_size_thres:
                            continue
                    person = {}
                    person['bbox'] = bbox[:5].cpu().numpy()
                    person_det_results.append(person)

            Q_person_det.put((p, im0, person_det_results, crop_paras, video_fps))
        except queue.Empty:
            print('Person detection done.')
            break

def intention_thread(model_pos, res_model, intent_model, play_audio, Q_target, Q_save_features=None):
    while True:
        # t0 = time.time()
        target_id, singleperson_track_dict, im0 = Q_target.get()
        # t1 = time.time()
        if Q_save_features is not None:
            if intent_model is not None:
                features = intention_inference(singleperson_track_dict, im0, model_pos, res_model, intent_model=intent_model, return_features=True, play_audio=play_audio)
                Q_save_features.put((target_id, features))
            else:
                features = intention_inference(singleperson_track_dict, im0, model_pos, res_model, intent_model=intent_model, return_features=True, play_audio=False)
                Q_save_features.put((target_id, features))
        else:
            intention_inference(singleperson_track_dict, im0, model_pos, res_model, intent_model=intent_model, play_audio=play_audio)

def sync_result_and_save(Q_save_features, Q_save_data, Q_missed_data, save_path):
    tracking_save_dir = {}
    while True:
        if Q_save_features.empty() and Q_save_data.empty() and Q_missed_data.empty():
            time.sleep(1)
        if not Q_save_features.empty():
            target_id, features = Q_save_features.get()
            if len(features) == 3:
                tracking_save_dir[target_id] = {'features':features}
            elif len(features) == 4:
                tracking_save_dir[target_id] = {'features':features[:3], 'prediction':features[3]}
            elif len(features) == 5:
                tracking_save_dir[target_id] = {'features':features[:3], 'prediction':features[3], 'is_played':features[4]}
            print('Tracking_ids:', tracking_save_dir.keys())
        elif not Q_missed_data.empty():
            tracking_id, person_dict, save_imgs, video_fps = Q_missed_data.get()
            if tracking_id in tracking_save_dir.keys():
                del tracking_save_dir[tracking_id]
                print(f'Data of #{tracking_id} has been missed.')
        elif not Q_save_data.empty():
            # save_in_one_file = (Q_save_data.qsize() > 1)
            save_in_one_file = False
            tracking_id, person_dict, save_imgs, video_fps = Q_save_data.get()
            if tracking_id in tracking_save_dir.keys():
                for k, v in person_dict.items():
                    tracking_save_dir[tracking_id][k] = v
                save_result_func(tracking_id, tracking_save_dir[tracking_id], save_path, save_imgs, video_fps, save_in_one_file)
                del tracking_save_dir[tracking_id]

def run(
        source='0',
        use_crop=False,
        webcam=False,
        use_mp=False,
        show_vid=True, # show realtime video using opencv
        crop_paras=None,
        save_result=False,
        save_path=None,
        skip=False,
        start_frame=0,
        start_id=0,
        cvt_color=False,
        play_audio=False,
        # yolo args
        # yolo_weights='./weights/yolov5m.pt',
        yolo_weights='yolov5/weights/crowdhuman_yolov5m.pt',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        # bbox_size_thres=80000,
        bbox_size_thres=83000,
        yolo_device='cuda:0',
        # 2d pose args
        # HrNet_W48
        # pose_detector_config='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py',
        # pose_detector_checkpoint='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
        # ResNetV1d-50
        pose_detector_config='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d50_coco_256x192.py',
        pose_detector_checkpoint='https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_256x192-a243b840_20200727.pth',
        use_multi_frames=False, # whether to use multi frames for inference in the 2D pose
        pose_device='cuda:0',
        # 3d pose args
        videopose_checkpoint='checkpoint',
        videopose_evaluate='pretrained_h36m_detectron_coco.bin',
        pose_3d_device='cuda:0',
        # ResNet args
        resnet_divice='cuda:0',
        # Intention args
        intention_model_type='cnn+mlp+mlp-double',
        intention_model_path='weights/model.pkl',
        intention_device='cuda:0',
):


    # Check source
    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or webcam is True
    if is_url and is_file:
        source = check_file(source)  # download

    # Load yolov5 model
    device = torch.device(yolo_device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load 2d pose model
    pose_detector_config = os.path.join(MMPOSE_PATH, pose_detector_config)

    pose_det_model = init_pose_model(
        pose_detector_config,
        pose_detector_checkpoint,
        device=pose_device)

    pose_det_dataset = pose_det_model.cfg.data['test']['type']
    dataset_info = pose_det_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # Load 3d poes model
    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], 
                            causal=False, # use causal convolutions for real-time processing
                            dropout=0.25, # dropout probability
                            channels=1024, # number of channels in convolution layers
                            dense=False # use dense convolutions instead of dilated convolutions
                            )

    model_pos = model_pos.to(pose_3d_device)

    videopose_checkpoint = os.path.join(VIDEO_POSE_PATH, videopose_checkpoint)
    chk_filename = os.path.join(videopose_checkpoint, videopose_evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])

    # Load ResNet
    res_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    res_model = res_model.to(resnet_divice)

    # Load intention model
    intention_model = MODEL[intention_model_type]([(651, 17, 3), (2048,), (2048,)],2)
    intention_model.load_state_dict(torch.load(intention_model_path))
    intention_model = intention_model.to(intention_device)

    # Yolo model warmup
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0, 0.0], 0

    # Multiprocessing start method reset.
    if mp.get_start_method() == 'fork' and use_mp:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method reset: {}".format(mp.get_start_method()))

    # Init Queue
    if use_mp:
        Q_dataset = mp.Queue(maxsize=20)
        Q_person_det = mp.Queue(maxsize=20)
        Q_target = mp.Queue()
        # Q_dataset = mp.Queue()
        # Q_person_det = mp.Queue()
    else:
        Q_dataset = Queue(maxsize=20)
        Q_person_det = Queue(maxsize=20)
        Q_target = Queue()
        # Q_im = Queue()

    # Init save
    if save_result:
        if use_mp:
            Q_save_features = mp.Queue(maxsize=20)
            Q_save_data = mp.Queue(maxsize=3)
            Q_missed_data = mp.Queue()
            process_sync_save = mp.Process(target=sync_result_and_save, args=(Q_save_features, Q_save_data, Q_missed_data, save_path), daemon=True)
        else:
            Q_save_features = Queue(maxsize=20)
            Q_save_data = Queue(maxsize=3)
            Q_missed_data = Queue()
            process_sync_save = Thread(target=sync_result_and_save, args=(Q_save_features, Q_save_data, Q_missed_data, save_path), daemon=True)
    else:
        Q_save_features = None

    # Multiprocessing start
    if use_mp:
        process_get_data = mp.Process(target=get_data_from_dataset, args=(source, webcam, imgsz, stride, cvt_color, pt, Q_dataset, device, half, use_crop, crop_paras, skip, start_frame), daemon=True)
        process_yolo_detect = mp.Process(target=person_detection_yolo, args=(model, Q_dataset, Q_person_det ,augment, visualize, conf_thres, iou_thres, classes, agnostic_nms, max_det, bbox_size_thres), daemon=True)
        process_intention_recog = mp.Process(target=intention_thread, args=(model_pos, res_model, intention_model, play_audio, Q_target, Q_save_features), daemon=True)
    else:
        process_get_data = Thread(target=get_data_from_dataset, args=(source, webcam, imgsz, stride, cvt_color, pt, Q_dataset, device, half, use_crop, crop_paras, skip, start_frame), daemon=True)
        process_yolo_detect = Thread(target=person_detection_yolo, args=(model, Q_dataset, Q_person_det, augment, visualize, conf_thres, iou_thres, classes, agnostic_nms, max_det, bbox_size_thres), daemon=True)
        process_intention_recog = Thread(target=intention_thread, args=(model_pos, res_model, intention_model, play_audio, Q_target, Q_save_features), daemon=True)
        # process_show_im = Thread(target=show_images_real_time, args=(Q_im, 10), daemon=True)

    process_get_data.start()
    print('Start dataload process.')
    process_yolo_detect.start()
    print('Start Yolo person detection process.')
    process_intention_recog.start()
    print('Start intention recognition process.')
    # process_show_im.start()
    if save_result:
        print('Start data saving process.')
        process_sync_save.start()

    start_time = 0

    # 3D pose estimation
    pose_det_results_list = []
    if skip:
        next_id = start_id
    else:
        next_id = 0
    pose_det_results = []

    person_tracking_dict = {}
    if save_result:
        im0_list = []

    tracked_ids = []
    target_ids = []

    while True:

        try:
            end_time = time.time()

            if save_result and not webcam:
                if skip:
                    # with open(os.path.join(save_path, 'last_frame.txt'), 'w') as f:
                    #     f.write(str(seen+start_frame) + ', ' + str(next_id))
                    with open(os.path.join(save_path, 'last_frame.pkl'), 'wb') as f:
                        pickle.dump([seen+start_frame, next_id], f)
                else:
                    # with open(os.path.join(save_path, 'last_frame.txt'), 'w') as f:
                    #     f.write(str(seen) + ', ' + str(next_id))
                    with open(os.path.join(save_path, 'last_frame.pkl'), 'wb') as f:
                        pickle.dump([seen, next_id], f)

            # print('Q_person_size:', Q_person_det.qsize())
            p, im0, person_det_results, new_crop_paras, video_fps = Q_person_det.get(timeout=10)
            # Q_im.put((p, im0))

            t1 = time.time()
            # read_times.append(t1-end_time)

            if start_time == 0:
                start_time = -1
            elif start_time == -1:
                start_time = time.time()
        
            # 2D pose estimation for current image
            pose_det_results_last = pose_det_results

            pose_det_results, _ = inference_top_down_pose_model(
                pose_det_model,
                im0,
                person_det_results,
                bbox_thr=0,
                format='xyxy',
                dataset=pose_det_dataset,
                dataset_info=dataset_info,
                return_heatmap=False, # whether to return heatmap
                outputs=None  # return the output of some desired layers, e.g. use ('backbone', ) to return backbone feature
            )

            # get track id for each person instance
            # pose_det_results (List[Dict{'keypoints':Array, 'bbox':Array, 'track_id':int}])
            pose_det_results, next_id = get_track_id(
                pose_det_results,
                pose_det_results_last,
                next_id,
                use_oks=False, # using OKS tracking
                tracking_thr=0.3 # tracking threshold
                )

            if save_result:
                existing_ids = [pose_det_result['track_id'] for pose_det_result in pose_det_results]
                not_existing_ids = []
                for tracking_id in target_ids:
                    if tracking_id not in existing_ids:
                        not_existing_ids.append(tracking_id)
                        if tracking_id in too_long_ids:
                            Q_missed_data.put((tracking_id, -1, -1, -1))
                        else:
                            save_imgs = im0_list[-len(person_tracking_dict[tracking_id]['bbox']):]
                            if Q_save_data.full():
                                if webcam:
                                    print('Saving process is busy now.')
                                    Q_missed_data.put((tracking_id, -1, -1, -1))
                                else:
                                    print('Waiting...')
                                    while Q_save_data.full():
                                        time.sleep(0.1)
                                    print(f'Put data of #{tracking_id}.')
                                    Q_save_data.put((tracking_id, person_tracking_dict[tracking_id].copy(), save_imgs, video_fps))
                            else:
                                print(f'Put data of #{tracking_id}.')
                                Q_save_data.put((tracking_id, person_tracking_dict[tracking_id].copy(), save_imgs, video_fps))
                            if skip:
                                with open(os.path.join(save_path, 'last_save_frame.pkl'), 'wb') as f:
                                    pickle.dump([seen+start_frame, next_id], f)
                            else:
                                with open(os.path.join(save_path, 'last_save_frame.pkl'), 'wb') as f:
                                    pickle.dump([seen, next_id], f)
                for not_existing_id in not_existing_ids:
                    target_ids.remove(not_existing_id)

            person_tracking_dict = update_person_tracking_dict(person_tracking_dict, pose_det_results)
            if save_result:
                im0_list, too_long_ids = update_imgs_list(im0_list, person_tracking_dict, im0)

            singleperson_track_dict, target_id, tracked_ids = pick_up_target(person_tracking_dict, new_crop_paras, tracked_ids)
            
            if target_id is not None:
                print('############### Tracking:', target_id)
                # intention_inference(singleperson_track_dict, model_pos, im0)
                Q_target.put((target_id, singleperson_track_dict, im0))
                if save_result:
                    target_ids.append(target_id)

            seen += 1

            if show_vid:
                annotator = Annotator(im0, line_width=2)
                for pose_det_result in pose_det_results:
                    id = int(pose_det_result['track_id'])
                    conf = (float(pose_det_result['bbox'][-1]) if len(pose_det_result['bbox'])==5 else None)
                    label = (f'id:{id} {conf:.2f}' if conf is not None else f'id:{id} person')
                    bboxes = pose_det_result['bbox'][:4]
                    annotator.box_label(bboxes, label, color=colors(0, True))

                    for point in pose_det_result['keypoints']:
                        annotator.im = cv2.circle(annotator.im, center=point[:2].astype(int), radius=5, color=(255,255,255), thickness=3)
                        annotator.im = cv2.circle(annotator.im, center=point[:2].astype(int), radius=4, color=(0,0,255), thickness=-1)

                annotator.im = cv2.line(annotator.im, (new_crop_paras[0],new_crop_paras[1]), (new_crop_paras[2],new_crop_paras[3]), color=(255,255,255), thickness=3)
                # Stream results
                im_show = annotator.result()
                cv2.imshow(str(p), im_show)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    print('Key interrupt.')
                    cv2.destroyAllWindows()
                    process_get_data.join(1)
                    process_yolo_detect.join(1)
                    process_intention_recog.join(1)
                    break

                print('showed.')

        except KeyboardInterrupt:
            print('Key interrupt.')
            cv2.destroyAllWindows()
            process_get_data.join(1)
            process_yolo_detect.join(1)
            process_intention_recog.join(1)
            print((end_time-start_time)/(seen-1)*30)
            exit()

        except queue.Empty:
            print('All frames processed.')
            cv2.destroyAllWindows()
            process_get_data.join()
            process_yolo_detect.join()
            process_intention_recog.join(1)
            break

    if save_result:
        end_t0 = time.time()
        if not Q_save_data.empty():
            print('Waiting for saving...')
            while time.time() - end_t0 < 60:
                if Q_save_data.empty():
                    print('Saved data processed.')
                    time.sleep(25)
                    break
                else:
                    time.sleep(1)
        else:
            time.sleep(25)
        process_sync_save.join(1)

    print((end_time-start_time)/(seen-1)*30)