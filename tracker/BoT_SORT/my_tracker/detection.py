import numpy as np

class Detection(object):
    '''
    Class này biểu diễn từng bounding box detect được trong 1 ảnh

    Parameters
    ----------
    tlwh : array_like
        Bounding box có dạng `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector của từng đối tượng có trong ảnh.
    frame_idx : int
        Id của frame đang được detect
    color_hist : array_like
        Một mảng lưu các thông tin histogram color của đối tượng detect được
    
    '''
    def __init__(self, tlwh, confidence, feature, frame_idx, color_hist=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.frame_idx = frame_idx
        self.color_hist = color_hist
    
    @property
    def tlbr(self):
        '''
        Chuyển tọa độ của bounding box từ (top left width height) thành (top left bottom right)
        tlwh -> tlbr
        '''
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xywh(self):
        '''
        Chuyển tọa độ của bounding box từ (top left width height) thành (x_center y_center width height)
        tlwh -> xywh (Dùng trong gate_cost_matrix() matching.py)
        '''
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret

