import os
import sys
import cv2
import argparse
from tqdm import tqdm
sys.path.append('../')
from configs import cfg

def ignore_region(img, region):
    if img is None:
        print("[Err]: Input image is none!")
        return -1
    img = img * (region > 0)
    return img

def preprocess(src_root, dst_root, sec):
    if not os.path.isdir(src_root):
        print("[Err]: Invalid source root")
        return

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)
        print(f"{dst_root} made")

    sec_dir_list = [sec]
    dst_dir_list = [dst_root + i for i in sec_dir_list]    # datasets/vid2img/train, test, val

    for i in dst_dir_list:
        if not os.path.isdir(i):
            os.makedirs(i)

    for idx, sec in enumerate(sec_dir_list):
        print(f"[INFO] Process {sec} folder...!")
        sec_path = src_root + '/' + sec
        if os.path.isdir(sec_path):
            for seq in os.listdir(sec_path):    # S01, S02,...
                if seq.startswith('S'):
                    seq_path = os.path.join(sec_path, seq)
                    for cam in os.listdir(seq_path):    # c001, c002,...
                        if cam.startswith('c'):
                            print(f"[INFO] Process camera {cam} in sequence {seq}....")
                            cam_path = os.path.join(seq_path, cam)
                            vid_path = os.path.join(cam_path, 'video.mp4')

                            dst_img_dir = os.path.join(dst_dir_list[idx], seq, cam, 'img1')
                            if not os.path.isdir(dst_img_dir):
                                os.makedirs(dst_img_dir)
                            
                            video = cv2.VideoCapture(vid_path)
                            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                            frame_current = 0
                          
                            while frame_current < frame_count - 1:
                                frame_current = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                                _, frame = video.read()
                                dst_f = 'img{:06d}.jpg'.format(frame_current)
                                dst_f_path = os.path.join(dst_img_dir, dst_f)
                                
                                if not os.path.isfile(dst_f_path):
                                    # frame = ignore_region(frame, ignor_region)
                                    cv2.imwrite(dst_f_path, frame)
                                    # print(f"{cam}: {dst_f} generated to {dst_img_dir}")
                                else:
                                    print(f"{cam}: {dst_f} already exists.")
                            print(f"[SUCCESS] Done camera {cam}")

if __name__ == '__main__':
    print("[PROCESS] Generate image from video...!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='mcmt_all.yml', help='load config')
    parser.add_argument('--sec', default='train', help="train/val/test part")
    opt = parser.parse_args()

    cfg.merge_from_file(f'../configs/{opt.config}')
    cfg.freeze()

    preprocess(src_root=cfg.CHALLENGE_DATA_DIR, dst_root=cfg.DET_SOURCE_DIR, sec=opt.sec)

    print("[Success] Done!")