"""Merge reid feature from different models."""

import os
import pickle
import sys
import argparse

from sklearn import preprocessing
import numpy as np
sys.path.append('../')
from configs import cfg


def merge_feat():
    """Save feature."""

    # NOTE: modify the ensemble list here
    ensemble_list = ['reid1', 'reid2']  # Số lượng backbone
    all_feat_dir = cfg.DATA_DIR.split('patches')[0]

    for cam in ['c001']:    # os.listdir()
        feat_dic_list = []
        for feat_mode in ensemble_list:
            data_path = os.path.join(all_feat_dir, feat_mode, opt.sec, opt.seq).replace("\\", "/")
            feat_pkl_file = os.path.join(data_path, cam, f'{cam}_dets_feat.pkl').replace("\\", "/")
            feat_mode_dic = pickle.load(open(feat_pkl_file, 'rb'))
            feat_dic_list.append(feat_mode_dic)
        merged_dic = feat_dic_list[0].copy()

        for patch_name in merged_dic:
            # print(patch_name)
            patch_feature_list = []
            for feat_mode_dic in feat_dic_list:
                patch_feature_list.append(feat_mode_dic[patch_name]['feat'])
            patch_feature_array = np.array(patch_feature_list)
            patch_feature_array = preprocessing.normalize(patch_feature_array,
                                                            norm='l2', axis=1)
            patch_feature_mean = np.mean(patch_feature_array, axis=0)
            merged_dic[patch_name]['feat'] = patch_feature_mean

        merge_dir = os.path.join(all_feat_dir, 'reid_merge', opt.sec, opt.seq, cam)
        if not os.path.exists(merge_dir):
            os.makedirs(merge_dir)

        merged_pkl_file = os.path.join(merge_dir, f'{cam}_dets_feat.pkl')
        pickle.dump(merged_dic, open(merged_pkl_file, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        print('save pickle in %s' % merged_pkl_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='mcmt_all.yml', help='load config')
    parser.add_argument('--sec', default='train', help="train/val/test part")
    parser.add_argument('--seq', default='S01', help="name of sequence")
    # parser.add_argument('--cam', default='c005', help="name of camera")
    opt = parser.parse_args()

    cfg.merge_from_file(f'../configs/{opt.config}')
    cfg.freeze()
    merge_feat()