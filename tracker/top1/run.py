import warnings
from os.path import join
warnings.filterwarnings("ignore")
from opts import opt
from deep_sort_app import run

if __name__ == '__main__':
    #print(opt)
    for i, cam in enumerate(opt.cams, start=1):
        print('processing the {}th video {}...'.format(i, cam))
        path_save = join(opt.dir_save, cam + '.txt').replace('\\', '/')    # ./scmt/cam/cam.txt
        run(
            # sequence_dir="G:\\My Drive\\AI CITY\\Code2\\datasets\\vid2img\\train\\S01\\c005",
            sequence_dir=join(opt.dir_dataset, cam).replace('\\', '/') ,
            # detection_file="G:\\My Drive\\AI CITY\\Code2\\datasets\\reid_merge\\train\\S01\\c005\\c005_dets_feat.pkl",
            detection_file=join(opt.dir_dets, cam, cam + '_dets_feat.pkl').replace('\\', '/') ,
            output_file=path_save,
            min_confidence=opt.min_confidence,
            nms_max_overlap=opt.nms_max_overlap,
            min_detection_height=opt.min_detection_height,
            max_cosine_distance=opt.max_cosine_distance,
            nn_budget=opt.nn_budget,
            display=True
        )