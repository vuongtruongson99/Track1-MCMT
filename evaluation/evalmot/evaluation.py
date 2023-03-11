import argparse
import copy
import os

import motmetrics as mm
import numpy as np

from tools import (
    mkdir_ifmiss,
    motchallenge_metric_names,
    read_results,
    save_dmexcel,
    unzip_objs,
)

mm.lap.default_solver = "lap"


class Evaluator(object):
    def __init__(self, data_root, seq_name):
        self.data_root = data_root
        self.seq_name = seq_name
        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', "gt.txt").replace("\\", "/")
        self.gt_frame_dict = read_results(gt_filename,"mot")
        self.gt_ignore_frame_dict = {}

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)
        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]
        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]
        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]
            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]
        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)
        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)
        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, "last_mot_events"):
            events = (
                self.acc.last_mot_events
            )  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()
        result_frame_dict = read_results(filename,"mot")
        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)
        return self.acc

    @staticmethod
    def get_summary(
        accs, names, metrics=("mota", "num_switches", "idp", "idr", "idf1", "precision", "recall")
    ):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)
        mh = mm.metrics.create()
        summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)
        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd

        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()

def run(opt):
    print(opt)
    seqs = sorted(os.listdir(opt.data_dir))
    pre_cams = sorted(os.listdir(opt.pre_fileroot))
    print(seqs, pre_cams)

    if len(seqs) != len(pre_cams):
        print('match error')
        return

    acc = []
    print("=" * 80)
    print("files:")
    print("=" * 80)
    for i, seq in enumerate(seqs):
        pre_path = os.path.join(opt.pre_fileroot, pre_cams[i], opt.mot_file_dir).replace("\\", "/")
        prefiles = os.listdir(pre_path)
        pre_file = [i for i in prefiles if opt.mot_file_type in i][0]

        evaluator = Evaluator(opt.data_dir, seq)

        pdt_path = os.path.join(pre_path, pre_file).replace("\\", "/")
        acc.append(evaluator.eval_file(pdt_path))
        print("gt:", os.path.join(opt.data_dir, seqs[i]), "   pdt:", pdt_path)
        # print(pre_path)

    print("=" * 80)
    print("summary")
    print("=" * 80)
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(acc, seqs, metrics)
    formatters, namemap = mh.formatters, motchallenge_metric_names
    strsummary = mm.io.render_summary(summary, formatters=formatters, namemap=namemap)
    print(strsummary)

# G:\My Drive\AI CITY\Code\datasets\AIC22_Track1_MTMC_Tracking\train\S01
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="G:\\My Drive\\AI CITY\\Code2\\datasets\\AIC22_Track1_MTMC_Tracking\\train\\S01", type=str, help="root of dataset")
    parser.add_argument("--pre_fileroot", default="G:\\My Drive\\AI CITY\\Code2\\datasets\\AIC22_Track1_MTMC_Tracking\\train\\S01", help="the prediction file")
    parser.add_argument("--mot_file_dir", default="mtsc", help="det result file path")
    parser.add_argument("--mot_file_type", default="mtsc_deepsort_yolov7", help="det result file type")

    parser.add_argument("--output_path", default="exptemdata", help="output path,eg excel file")
    parser.add_argument("--excel_file", default="save_gt1.xlsx", help="saved excel file")
    opt = parser.parse_args()
    run(opt)