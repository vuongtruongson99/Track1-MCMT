import os
import argparse
from natsort import natsorted
from tqdm import tqdm
import cv2

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255,  0), (0, 128,  0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139,  0,
                                                               139), (100, 149, 237), (138, 43, 226), (238, 130, 238),
             (255,  0, 255), (0, 100,  0), (127, 255,  0), (255,  0,
                                                            255), (0,  0, 205), (255, 140,  0), (255, 239, 213),
             (199, 21, 133), (124, 252,  0), (147, 112, 219), (106, 90,
                                                               205), (176, 196, 222), (65, 105, 225), (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199,
                                                               21, 133), (148,  0, 211), (255, 99, 71), (144, 238, 144),
             (255, 255,  0), (230, 230, 250), (0,  0, 255), (128, 128,
                                                             0), (189, 183, 107), (255, 255, 224), (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128,
                                                               128), (72, 209, 204), (139, 69, 19), (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135,
                                                               206, 235), (0, 191, 255), (176, 224, 230), (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139,
                                                                 139), (143, 188, 143), (255,  0,  0), (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42,
                                                              42), (178, 34, 34), (175, 238, 238), (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--dw", type=int, default=1920)
    parser.add_argument("--dh", type=int, default=1080)
    parser.add_argument("--save_path", type=str, default="demo.avi")

    return parser.parse_args()

def draw_bbox(img, box, cls_name, identity=None, offset=(0, 0)):
    '''
        draw box of an id
    '''
    x1, y1, x2, y2 = [int(i+offset[idx % 2]) for idx, i in enumerate(box)]
    # set color and label text
    color = COLORS_10[identity %
                      len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = '{} {}'.format(cls_name, identity)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
    cv2.putText(img, label, (x1, y1+t_size[1]+4),
                cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img


def draw_bboxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(
            img, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


if __name__ == '__main__':
    args = parse_args()

    imgs = natsorted(os.listdir(args.video_path))
    imgs = [os.path.join(args.video_path, img).replace('\\', '/') 
            for img in imgs if img.endswith('.jpg')]

    gt = open(os.path.join(args.gt_path))

    gt_dict = dict()

    for line in gt.readlines():
        fid, pid, x, y, w, h, flag = [int(float(a)) for a in line.strip().split(',')][:7]
        
        if fid not in gt_dict:
            gt_dict[fid] = [(pid, x, y, w, h)]
        else:
            gt_dict[fid].append((pid, x, y, w, h))
    # print(gt_dict)

    print(os.path.basename(args.gt_path)[:-4] + '.mp4')

    for img in imgs:
        ori_im = cv2.imread(img)
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
        fid = int(img[-8:-4])
        print(fid)
        bbox_xyxy = []
        pids = []
        if fid not in gt_dict:
            continue
        for pid, x, y, w, h in gt_dict[fid]:
            bbox_xyxy.append([x, y, x+w, y+h])
            pids.append(pid)

        ori_im = draw_bboxes(ori_im, bbox_xyxy, pids)
        cv2.putText(ori_im, "%d" % (fid), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 2)
        cv2.imshow('image',ori_im)
        cv2.waitKey(0)

# python .\visual_general_result.py --video_path "G:\My Drive\AI CITY\Code2\datasets\results\detection\images\train\S01\c005\img1" --gt_path "G:\My Drive\AI CITY\Code2\datasets\AIC22_Track1_MTMC_Tracking\train\S01\c005\gt\gt.txt"