import argparse
import time
from pathlib import Path
import os
import numpy as np
import pickle

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

import sys
sys.path.append('../../')
from configs import cfg


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        
    # Directories: cfg['DATA_DIR]/train/S01/c005
    save_dir = Path(Path(opt.project) / Path(opt.sec) / Path(opt.seq) / Path(opt.cam))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model.stride = 8, 16, 32
    imgsz = check_img_size(imgsz, s=stride)  # kiểm tra size có là bội của stride không
    # if trace:
    #     model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    out_dict = dict()
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path    
            save_path = str(save_dir / 'dets_debug' / p.name)  # cfg['DATA_DIR]/train/S01/c005/dets_debug/img000000.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            det_path = str(save_dir / 'dets' / p.stem)

            if not os.path.isdir(str(save_dir / 'dets')):
                os.makedirs(str(save_dir/'dets'))
            if not os.path.isdir(str(save_dir / 'dets_debug')):
                os.makedirs(str(save_dir/'dets_debug'))
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                img_det = np.copy(im0)

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                det_num = 0
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    x1,y1,x2,y2 = tuple(torch.tensor(xyxy).view(4).tolist())
                    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

                    # Clip bbox
                    if x1 < 0 or y1 < 0 or x2 > im0.shape[1]-1  or y2 > im0.shape[0]-1:
                        continue

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    # Tách từng object detect được rồi lưu thành 1 file để train reid
                    if True:
                        det_name = p.stem + "_{:0>3d}".format(det_num)              # img000000_000, img000000_001,...
                        det_img_path = det_path + "_{:0>3d}.jpg".format(det_num)    # /S01/c005/dets/img000001_013.jpg: lưu patch của object detect được trong 1 frame
                        det_class = int(cls.tolist())                               # 2, 5, 7: các class được detect
                        det_conf = conf.tolist()
                        cv2.imwrite(det_img_path, img_det[y1:y2, x1:x2])
                        out_dict[det_name] = {
                            'bbox': (x1, y1, x2, y2),
                            'frame': p.stem,
                            'id': det_num,
                            'imgname': det_name + '.jpg',
                            'class': det_class,
                            'conf': det_conf
                        }
                    
                    det_num += 1

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow("test", im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        pickle.dump(out_dict, open(str(save_dir/'{}_dets.pkl'.format(opt.cam)), 'wb'))

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')                # folder of imgs
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')    
    parser.add_argument('--cfg_file', default='mcmt_all.yml', help='config file')
    parser.add_argument('--sec', default='train', help="train/val/test part")
    parser.add_argument('--seq', default='S01', help="name of sequence")
    parser.add_argument('--cam', default='c001', help='save results to project/cam')
    opt = parser.parse_args()

    cfg.merge_from_file(f'../../configs/{opt.cfg_file}')
    cfg.freeze()

    opt.project = cfg.DATA_DIR                                                  # datasets/patches/     output folder
    opt.source = f"{cfg.DET_SOURCE_DIR}/{opt.sec}/{opt.seq}/{opt.cam}/img1"    # input folder

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt', 'yolov7-e6e.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
