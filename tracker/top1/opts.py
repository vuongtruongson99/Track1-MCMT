"""
@Filename: opts.py
@Discription: opts
"""
import json
import argparse
from os.path import join

data = {
    'train': {
        'S04': [
            'c001'
        ]
    },
    'test': {
        'S06': [
            'c041',
            'c042',
            'c043',
            'c044',
            'c045',
            'c046'
        ]
    }
}

# data = {
#     'AICITY': {
#         'test':[
#             'c041',
#             'c042',
#             'c043',
#             'c044',
#             'c045',
#             'c046'
#         ],
#         'train': [
#             'c001',
#             'c002',
#             'c003',
#             'c004',
#             'c005'
#         ]
#     }
# }

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            'mode',
            type=str,
            default='train',
        )
        self.parser.add_argument(
            'seq',
            type=str,
            default='S01',
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter',
            default=True
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism',
            default=True
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost',
            default=True
        )
        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching',
            default=True
        )
        self.parser.add_argument(
            '--root_dataset',
            default='G:\\My Drive\\AI_CITY\\Code2\\datasets\\'      # Thay đổi đường dẫn vào folder ~/datasets/
        )
        self.parser.add_argument(
            '--dir_save',
            default='./scmt'
        )
        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            default=0.98
        )

    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.min_confidence = 0.1
        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        opt.max_cosine_distance = 0.4
        opt.dir_dets = join(opt.root_dataset, 'reid_merge', opt.mode, opt.seq)      # datasets/reid_merge/train/S01/
   
        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        opt.cams = data[opt.mode][opt.seq]     # c001, c002, c003, c004, c005
        opt.dir_dataset = join(                     # ~/datasets/vid2img/train/S01
            opt.root_dataset,
            'vid2img',
            opt.mode,
            opt.seq
        )
        return opt

opt = opts().parse()