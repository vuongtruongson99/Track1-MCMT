# Single Camera Vehicle Tracking 
---
## To do task 
- [ ] Tạo pipeline train reid
---

## Requirements:
```
pip install -r requirements.txt
```
---
## Chuẩn bị dữ liệu:

- Tải dataset từ track 1 AIC2022 và đặt toàn bộ dataset bên trong folder `datasets`. 
- Tải pre-trained models `yolov7-e6e` của Yolov7 từ [GitHub](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)
- Tải pre-trained models từ [GoogleDrive](https://drive.google.com/drive/folders/1AwVib1W9K-rOB4SFEgTT8d7moCK4F6QD?usp=share_link) và đặt nó vào trong folder `reid/reid_model/`

Cấu trúc như sau:

```
├── Track1-MCMT
    ├── datasets
    │   └── AIC22_Track1_MTMC_Tracking
    │        ├── train
    │        ├── test    
    │        ├── validation 
    │        └── ...
    ├── detector
    |   └── yolov7 
    |       └── weights
    |           └── yolov7-e6e.pt
    └── reid
        └── reid_model
            ├── resnet101_ibn_a_2.pth
            ├── resnet101_ibn_a_3.pth
            └── resnext101_ibn_a_2.pth
```

Sau đó vào trong folder `configs` để thay đổi các đường dẫn absolute:


- File `configs/mcmt_all.yml`:
```
CHALLENGE_DATA_DIR: '/xxx/Track1-MCMT/datasets/AIC22_Track1_MTMC_Tracking/'     # data of contest
DET_SOURCE_DIR: '/xxx/Track1-MCMT/datasets/vid2img/'                            # tách các video thành từng frame ảnh
DATA_DIR: '/xxx/Track1-MCMT/datasets/patches'                                   # ảnh các xe sau khi đi qua detector được cắt ra
REID_SIZE_TEST: [384, 384]    # 384, 256
USE_RERANK: True
USE_FF: True
SCORE_THR: 0.1
MCMT_OUTPUT_TXT: 'track1.txt'
```

- File `configs/mcmt_reid1.yml`, `configs/mcmt_reid2.yml`, `configs/mcmt_reid3.yml`:
```
DET_SOURCE_DIR: 'xxx/Track1-MCMT/datasets/vid2img/'               # after split video into frame
REID_MODEL: 'reid_model/resnet101_ibn_a_2.pth'                    # pretrained weight
REID_BACKBONE: 'resnet101_ibn_a_2'                                # name of backbone
DET_IMG_DIR: 'xxx/Track1-MCMT/datasets/patches'                   # patches image
DATA_DIR: 'xxx/Track1-MCMT/datasets/reid1/'                       # save result
REID_SIZE_TEST: [384, 384]
```

- Rồi sau đó sẽ chạy lệnh
```shell
bash ./run_all.sh

```

*Lưu ý*: Để rút gọn thời gian, chỉ sử dụng sequence S01 trong `train`. 
---
