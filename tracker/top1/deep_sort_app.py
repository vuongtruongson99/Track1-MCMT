from __future__ import division, print_function, absolute_import

import argparse
import os
import copy
from tqdm import tqdm

import cv2
import numpy as np
import pickle

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from opts import opt


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
    img_dir = os.path.join(sequence_dir, "img1").replace('\\', '/') 

    img_filenames = {
        int(os.path.splitext(f)[0][3:]): os.path.join(img_dir, f) for f in os.listdir(img_dir) if 'desktop.ini' not in f
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
    cam = sequence_dir.split("/")[-1]
    pickle.dump(seq_info, open(f"color_hist_{cam}.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0, frame_img=None):
    """Create detections for given frame index from the raw detection matrix.
    """
    detection_list = []
    
    if frame_idx in detection_mat[0]:
        for i in range(len(detection_mat[0][frame_idx])):
            tlbr = detection_mat[0][frame_idx][i][:4]
            bbox = np.asarray(tlbr, dtype=np.float32).copy()
            bbox[2:] -= bbox[:2]

            if bbox[3] < min_height:
                continue
            confidence = detection_mat[0][frame_idx][i][4]
            feature = detection_mat[1][frame_idx][i]
            color_hist = detection_mat[2][frame_idx][i]

            detection_list.append(Detection(bbox, confidence, feature, frame_idx, color_hist=color_hist))
    
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.
    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    """
    print(sequence_dir, detection_file, output_file)
    # seq_info = gather_sequence_info(sequence_dir, detection_file)
    cam = sequence_dir.split("/")[-1]   # camera name from sequence_dir
    seq_info = pickle.load(open('color_hist_c001.pkl', 'rb'))
    # print(seq_info)
    # sequence = "c005"
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )

    tracker = Tracker(metric, cam_name=cam, image_filenames=seq_info['image_filenames'])
    results = []
    

    def frame_callback(vis, frame_idx):
        # Load image and generate detections.
        img = cv2.imread(seq_info["image_filenames"][frame_idx])
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height, frame_img=img)
        
        
        detections = [d for d in detections 
            if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)

        detections = [detections[i] for i in indices]

        # for d in detections:
        #     print(d.tlwh, d.confidence)

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

    def batch_callback():
        tracker.postprocess()
        # pass

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
        # visualizer = visualization.NoVisualization(seq_info)
    else:
        visualizer = visualization.NoVisualization(seq_info)

    # if cam in ['c044']:
    #     visualizer.run_reverse(frame_callback, batch_callback)
    # else:
    #     visualizer.run(frame_callback, batch_callback)
    visualizer.run(frame_callback)

    # if cam in ['c042']:
    #     res1 = copy.deepcopy(tracker.tracks_all)
    #     tracker = Tracker(metric, cam_name=cam, mask=roi_mask, image_filenames=seq_info['image_filenames'])
    #     visualizer.run_reverse(frame_callback, batch_callback)
    #     for track1 in tracker.tracks_all:
    #         if track1.storage[0].tlwh[0] > 600 and track1.storage[-1].tlwh[0] < 650 and track1.storage[0].tlwh[0] > track1.storage[-1].tlwh[0]:
    #             track1_frames = [d.frame_idx for d in track1.storage]
    #             for track2 in res1:
    #                 if track2.storage[0].tlwh[0] > 600 and track2.storage[-1].tlwh[0] < 650 and track2.storage[0].tlwh[0] > track1.storage[-1].tlwh[0]:
    #                     track2_frames = [d.frame_idx for d in track2.storage]
    #                     same_frame = set(track1_frames) & set(track2_frames)
    #                     if len(same_frame) > 7:
    #                         same_count = 0
    #                         for fid in same_frame:
    #                             det1 = [d for d in track1.storage if d.frame_idx == fid]
    #                             det2 = [d for d in track2.storage if d.frame_idx == fid]
    #                             if tracker._det_iou(det1[0], det2[0]) > 0.95:
    #                                 same_count += 1
    #                         if same_count > 5:
    #                             for det2 in track2.storage:
    #                                 if det2.frame_idx > track1.storage[-1].frame_idx:
    #                                     track1.storage.append(det2)

    # if cam in ['c044']:
    #     res1 = copy.deepcopy(tracker.tracks_all)
    #     tracker = Tracker(metric, cam_name=cam, mask=roi_mask, image_filenames=seq_info['image_filenames'])
    #     visualizer.run(frame_callback, batch_callback)
    #     for track1 in tracker.tracks_all:
    #         if track1.storage[0].tlwh[0] > 570 and track1.storage[-1].tlwh[0] < 570:
    #             track1_frames = [d.frame_idx for d in track1.storage]
    #             for track2 in res1:
    #                 if track2.storage[0].tlwh[0] > 570 and track2.storage[-1].tlwh[0] < 570:
    #                     track2_frames = [d.frame_idx for d in track2.storage]
    #                     same_frame = set(track1_frames) & set(track2_frames)
    #                     if len(same_frame) > 6:
    #                         same_count = 0
    #                         for fid in same_frame:
    #                             det1 = [d for d in track1.storage if d.frame_idx == fid]
    #                             det2 = [d for d in track2.storage if d.frame_idx == fid]
    #                             if tracker._det_iou(det1[0], det2[0]) > 0.95:
    #                                 same_count += 1
    #                         if same_count > 5:
    #                             for det2 in track2.storage:
    #                                 if det2.frame_idx < track1.storage[0].frame_idx:
    #                                     track1.storage.append(det2)
    #                             track1.storage.sort(key=lambda d : d.frame_idx)

    # dict_cam={} # dict có dạng: {'c005_1_1': [feature, color_hist]}
    # # Store results.
    # for track in tracker.tracks_all:
    #     for det in track.storage:
    #         bbox = det.tlwh
    #         if cam == 'c042' and (roi_mask[int(bbox[1] + bbox[3] / 2), int(bbox[0] + bbox[2] / 2), 0] / 255 != 1):
    #             continue
    #         frame_idx = det.frame_idx
    #         key = '{}_{}_{}'.format(cam, frame_idx + 1, track.track_id)
    #         if key in dict_cam:
    #             continue
    #         dict_cam[key] = np.hstack((det.feature, det.color_hist))
    #         results.append([
    #                 frame_idx + 1, track.track_id, round(bbox[0]), round(bbox[1]), round(bbox[2]), round(bbox[3])])

    # with open(output_file, 'w') as f:
    #     for row in sorted(results):
    #         print('%d,%d,%d,%d,%d,%d,1,-1,-1,-1' % (
    #             row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

    # with open(output_file.replace('.txt', '.pkl'), 'wb') as fid:
    #     pickle.dump(dict_cam, fid, protocol=2)

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.sequence_dir = "G:\\My Drive\\AI CITY\\Code2\\datasets\\vid2img\\train\\S01\\c005"
    args.detection_file = "G:\\My Drive\\AI CITY\\Code2\\datasets\\reid_merge\\train\\S01\\c005\\c005_dets_feat.pkl"
    args.output_file = "G:\\My Drive\\AI CITY\\Code2\\datasets\\AIC22_Track1_MTMC_Tracking\\train\\S01\\c005\\mtsc\\mtsc_deepsort_yolov7.txt"
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)