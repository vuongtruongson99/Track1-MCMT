import os
import cv2
import torch
import pickle
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from torchvision.ops import nms

from application_util import visualization
from application_util.visualization import create_unique_color_uchar
from fm_tracker.mc_bot_sort import BoTSORT

import sys
sys.path.append("../../")
from configs import cfg
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def gather_sequence_info(sequence_dir, detection_file, max_frame=0):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).
    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.
    Returns
    -------
    Dict
        A dictionary of the following sequence information:
        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.
    """
    img_dir = os.path.join(sequence_dir, 'img1')

    # dict: idx_frame - path to frame
    img_filenames = {
        int(os.path.splitext(f)[0][3:]): os.path.join(img_dir, f) for f in os.listdir(img_dir)
    }

    if len(img_filenames) > 0:
        image = cv2.imread(next(iter(img_filenames.values())), cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(img_filenames) > 0:
        min_frame_idx = min(img_filenames.keys())
        max_frame_idx = max(img_filenames.keys())
    if max_frame > 0:
        max_frame_idx = max_frame

    feature_dim = 2048
    det_feat_dic = pickle.load(open(detection_file, 'rb'))
    bbox_dic = {}
    feat_dic = {}
    ############ Thêm color hist ############
    color_hist_dic = {}
    #########################################

    # 1 frame sẽ có nhiều car
    # gộp các bbox và feat của từng car vào thành 1 frame_idx
    for image_name in tqdm(det_feat_dic):
        frame_idx = image_name.split("_")[0]
        frame_idx = int(frame_idx[3:])
        det_bbox = np.array(det_feat_dic[image_name]['bbox']).astype('float32')
        det_feat = det_feat_dic[image_name]['feat']
        score = det_feat_dic[image_name]['conf']
        score = np.array((score,))
        det_bbox = np.concatenate((det_bbox, score)).astype('float32')

        #################################### Thêm color hist ################################################
        img = cv2.imread(os.path.join(img_dir, det_feat_dic[image_name]['frame'] + '.jpg'))

        if img is not None:
            color_hist = []
            H, W, _ = img.shape
            x1 = int(det_bbox[0])
            y1 = int(det_bbox[1])
            x2 = int(det_bbox[2])
            y2 = int(det_bbox[3])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W - 1, x2)
            y2 = min(H - 1, y2)
            for i in range(3):
                color_hist += cv2.calcHist([img[y1:y2, x1:x2]], [i], None, [8], [0.0, 255.0]).T.tolist()[0]
            color_hist = np.array(color_hist)
            norm = np.linalg.norm(color_hist)
            color_hist /= norm
        #######################################################################################################
            
        if frame_idx not in bbox_dic:
            bbox_dic[frame_idx] = [det_bbox]
            feat_dic[frame_idx] = [det_feat]
            color_hist_dic[frame_idx] = [color_hist]
        else:
            bbox_dic[frame_idx].append(det_bbox)
            feat_dic[frame_idx].append(det_feat)
            color_hist_dic[frame_idx].append(color_hist)

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": img_filenames,
        "detections": [bbox_dic, feat_dic, color_hist_dic],
        "groundtruth": None,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": None,
        "frame_rate": 10
    }

    pickle.dump(seq_info, open("color_hist.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
    # return seq_info

def run(sequence_dir, detection_file, output_file, min_box_area=1000, display=False):
    # seq_info = gather_sequence_info(sequence_dir, detection_file, -1)
    seq_info = pickle.load(open('color_hist.pkl', 'rb'))

    tracker = BoTSORT(image_filenames=seq_info['image_filenames'])
    results = []
   
    if not output_file:
        return
    path = os.path.dirname(output_file)
    if not os.path.exists(path):
        os.makedirs(path)

    save_format = "{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n"

    def frame_callback(vis, frame_idx):
        img_path = seq_info['image_filenames'][frame_idx]
        save_path = img_path.replace('img1', 'img2')
        if not os.path.exists(save_path.split('img2')[0] + 'img2'):
            os.makedirs(save_path.split('img2')[0] + 'img2')
        
        img = cv2.imread(img_path)

        [bbox_dic, feat_dic, color_hist_dic] = seq_info['detections']
        if frame_idx not in bbox_dic:
            print(f'Empty for {frame_idx}')
            return
        
        detections_ori = bbox_dic[frame_idx]        # Tọa độ tất cả bbox trong 1 frame
        feats_ori = feat_dic[frame_idx]             # Feature vector ứng với từng bbox đó
        color_hist_ori = color_hist_dic[frame_idx]  # Color hists ứng với từng bbox trong đó 
        detections, feats, color_hists = [], [], []

        # for i, det in enumerate(detections_ori):
        #     if (det[2] - det[0]) * (det[3] - det[1]) > min_box_area:
        #         detections.append(det)
        #         feats.append(feats_ori[i])

        # boxes = np.array([d[:4] for d in detections], dtype=float)
        # scores = np.array([d[4] for d in detections], dtype=float)

        # # if boxes.ndim == 1:
        # #     boxes = boxes[np.newaxis, :]
        # #     scores = scores[np.newaxis, :]
        # nms_keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold=0.99).numpy()

        # detections = np.array([detections[i] for i in nms_keep], dtype=float)
        # feats = np.array([feats[i] for i in nms_keep], dtype=float)
        # # print(np.array(detections).shape)
        # # print(detections[0])

        detections = np.array(detections_ori, dtype=float)
        feats = np.array(feats_ori, dtype=float)
        color_hists = np.array(color_hist_ori, dtype=float)

        # print(detections)
        # Update tracker
        online_targets = tracker.update(detections, feats, color_hists, frame_idx, img)
        # online_targets = tracker.update1(detections, feats, frame_idx)

        cv2.putText(img, str(frame_idx), (10, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)

        # for t in online_targets:
        #     tlwh = t.det_tlwh
        #     tid = t.track_id
        #     score = t.score
        #     trk_color = create_unique_color_uchar(tid)

        #     if tlwh[2] * tlwh[3] > min_box_area:
        #         cv2.putText(img, str(tid) + '|' + str(round(score,2)), (int(tlwh[0]), int(tlwh[1]) - 4), cv2.FONT_HERSHEY_PLAIN, 1, trk_color, 2)
        #         cv2.rectangle(img, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), trk_color, 2)
        #         results.append([
        #             frame_idx + 1, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], score
        #         ])
        

        if display:
            vis.set_image(img.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(online_targets)
                 
    def batch_callback():
        tracker.postprocess()

    if display:
    # visualizer = visualization.Visualization(seq_info, update_ms=10)
        visualizer = visualization.NoVisualization(seq_info)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    
    visualizer.run(frame_callback, batch_callback)



    # with open(output_file, "w") as f:
    #     for result in results:
    #         line = save_format.format(
    #             frame=result[0], id=result[1], x1=result[2], y1=result[3], w=result[4], h=result[5], 
    #         )
    #         f.write(line)

if __name__ == '__main__':
    sequence_dir = "G:\\My Drive\\AI CITY\\Code2\\datasets\\vid2img\\train\\S01\\c005"
    detection_file = "G:\\My Drive\\AI CITY\\Code2\\datasets\\reid_merge\\train\\S01\\c005\\c005_dets_feat.pkl"
    output_file = "G:\\My Drive\\AI CITY\\Code2\\datasets\\AIC22_Track1_MTMC_Tracking\\train\\S01\\c005\\mtsc\\mtsc_botsort_yolov7.txt"

    run(sequence_dir, detection_file, output_file)

    # gather_sequence_info(sequence_dir, detection_file)
