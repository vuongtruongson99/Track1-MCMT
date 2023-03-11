import os, sys, pdb
import numpy as np
import pickle
import random
from tqdm import tqdm

#######################################################################
# Kiểm tra lại điều kiện cho từng cam, tại sao chọn y_max cho từng cam
def is_valid_position(x, y, w, h, cam_id):
    """
    To judge if the position is out of the image
    Args:
        None
    """
    x_max = 1280
    y_max = 720 if cam_id == 45 or cam_id == 46 else 960
    x2 = x + w
    y2 = y + h
    if x < 0 or y < 0 or x2 >= x_max or y2 >= y_max:
        return False
    return True
#######################################################################

def process_border(x, y, w, h, cam_id):
    x_max = 1280
    y_max = 720 if cam_id == 45 or cam_id == 46 else 960

    dw, dh = 0, 0
    if x < 1:
        dw = -x
        x = 1
    if y < 1:
        dh = -y
        y = 1
    x2, y2 = x + w, y + h
    w = x_max - x if x2 >= x_max else w - dw
    h = y_max - y if y2 >= y_max else h - dh
    return (x, y, w, h)


# src_root: data/track_results/scmt
def track_file_format_transfer(src_root, dst_root):
    """ Transfer file format of track results.
    Single camera format (the direct output file of single camera algorithm) -> Multi camera format (the submission format)
    All files must be named as "c04x.txt"
    Args:
        src_root:
        dst_root:
    """

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    cam_list = os.listdir(src_root)
    cam_list.sort()
    for cam_file in cam_list:
        print("[INFOR] Processing: {}".format(cam_file))
        cam_id = int(cam_file[1:4])  # c001.txt -> 1
        dst_obj = open(os.path.join(dst_root, cam_file).replace("\\", '/'), 'w')
        f_dict = {}
        with open(os.path.join(src_root, cam_file), 'r') as fid:
            for line in fid.readlines():
                s = [int(float(x)) for x in line.rstrip().split(',')]   # [frame_id, track_id, t, l, w, h]
                t, l, w, h = s[2: 6]
                if not is_valid_position(t, l, w, h, cam_id):
                    t, l, w, h = process_border(t, l, w, h, cam_id)
                    if w <= 0 or h <= 0:
                        continue
                    s[2:6] = t, l, w, h
                fr_id = s[0]
                line = '{} {} {} {} -1 -1\n'.format(cam_id, s[1], s[0], ' '.join(map(str, s[2:6]))) # [cam_id, track_id, frame_id, t, l, w, h, -1, -1]
                if fr_id not in f_dict:
                    f_dict[fr_id] = []
                f_dict[fr_id].append(line)
            
            fr_ids = sorted(f_dict.keys())
            for fr_id in fr_ids:
                for line in f_dict[fr_id]:
                    dst_obj.write(line)
        dst_obj.close()