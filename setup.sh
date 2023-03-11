pip install gdown

# Yolo model
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
mv yolov7-e6e.pt detector/yolov7/weights/

# ReId model 
gdown --id 1xN-MBRqZQ6JvxZkEvPo_hwGlPNVEU60M    # resnet101_ibn_a_2.pth
gdown --id 1i14Od3VA6FnO1kJpEvgzkPcr0NjBkUMa    # resnext101_ibn_a_2.pth

mv resnet101_ibn_a_2.pth reid/reid_model
mv resnext101_ibn_a_2.pth reid/reid_model