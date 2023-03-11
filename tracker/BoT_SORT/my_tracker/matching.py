import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q

def ious(atlbrs, btlbrs):
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious
 
def iou_distance(tracks, detections):
    """
    IoU metric
    
    Parameters
    ----------
    tracks : List[top1_tracker.Tracker.Tracker]
        List các tracks
    detection : List[top1_tracker.detection.Detection]
        List các detections
    """
    if (len(tracks)>0 and isinstance(tracks[0], np.ndarray)) or (len(detections) > 0 and isinstance(detections[0], np.ndarray)):
        track_tlbrs = tracks
        det_tlbrs = detections
    else:
        track_tlbrs = [track.tlbr for track in tracks]
        det_tlbrs = [det.tlbr for det in detections]

    _ious = ious(track_tlbrs, det_tlbrs)
    cost_matrix = 1 - _ious
    
    return cost_matrix

####### NOTE: Nên finetune lại metric #########
def embedding_distance(tracks, detections, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)

    if cost_matrix.size == 0:
        return cost_matrix

    det_features = [det.feature for det in detections]
    track_features = np.asarray([track.smooth_feature for track in tracks], dtype=np.float)

    ##################### Mode nhiều loại distance ###################
    if metric == 'fusion':
        metric_pool = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule']
        lambda_pool = np.random.rand(22)
        lambda_pool /= np.linalg.norm(lambda_pool) # Khi làm thật sẽ điều chỉnh weight cho đúng
        for i in range(len(metric_pool)):
            cost_matrix += lambda_pool[i]*cdist(track_features, det_features, metric_pool[i])
        cost_matrix = np.maximum(0.0, cost_matrix[i, :])
    ##################################################################
    else:
        cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix

##################### gate_cost_matrix nhưng cùng với Motion Camera #####################
def gate_cost_matrix(cost_matrix, tracks, detections, track_inds, detection_ids, 
                     gating_threshold=50, only_position=False, add_identity=True, lambda_=0.98):

    if cost_matrix.size == 0:
        return cost_matrix

    gating_threshold = gating_threshold
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([detections[i].to_xywh() for i in detection_ids])

    for row, track_id in enumerate(track_inds):
        track = tracks[track_id]
        gating_distance = track.kf.gating_distance(track.mean, track.covariance, measurements, only_position, add_identity=add_identity)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix
#################################################################################################

def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost

#################################### Có cần thiết hay không? ####################################
def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.confidence for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
#################################################################################################

def linear_assignment(cost_matrix, thresh):
    '''
    matches: [[index of track, index of detection]]
    unmatched_a (unmatch_track): xác định cột mà hàng được gán. Ví dụ x = [2, 0, 1] => Cột 2 gán với hàng 0, cột 0 gán với hàng 1, cột 1 gán với hàng 2
    unmatched_b (unmatch_detect): xác định hàng mà cột theo ind của mảng được gán. Ví dụ: y = [1, 0, 2] => hàng 1 gán với cột 0, hàng 0 gán với cột 1, hàng 2 gán với cột 2
    '''
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def min_cost_matching(distance_metric, match_thresh, tracks, detections,
                      track_inds=None, detection_inds=None):
    
    if track_inds is None:
        track_inds = np.arange(len(tracks))
    if detection_inds is None:
        detection_inds = np.arange(len(detections))

    if len(detection_inds) == 0 or len(track_inds) == 0:
        return [], track_inds, detection_inds

    cost_matrix = distance_metric(tracks, detections, track_inds, detection_inds)
    
    cost_matrix_ = cost_matrix.copy()

    matches, unmatched_tracks, unmatched_detection = linear_assignment(cost_matrix_, match_thresh)

    return matches, unmatched_tracks, unmatched_detection
    

def matching_cascade(distance_metric, match_thresh, cascade_depth, tracks, detections, 
                        track_inds=None, detection_inds=None, type='vanilla'):
    '''
    Thực hiện matching cascade

    Parameters
    ----------
    distance_metric : Callable(List[Track], List[Detection], List[int], List[int]) -> ndarray
        Hàm để tính distance metric giữa Track và Detection. Trả về cost matrix với phần 
        tử (i, j) ứng với cost giữa i-th Track và j-th Detection
    match_thresh : float
        CHƯA BIẾT ĐỂ LÀM GÌ :V
    cascade_depth : int
        Độ sâu của cascade, nên bằng với track age
    tracks : List[mc_bot_sort2.Track]
        List các predicted tracks tại frame hiện tại

    '''
    if track_inds is None:
        track_inds = list(range(len(tracks)))
    if detection_inds is None:
        detection_inds = list(range(len(detections)))

    unmatched_detections = detection_inds
    matches = []

    if type == 'vanilla':
        track_inds_l = [k for k in track_inds]
        matches_l, _, unmatched_detections = min_cost_matching(distance_metric, match_thresh, tracks, detections,
                                                                track_inds_l, unmatched_detections)
        
        matches += list(matches_l)

    elif type == 'cascade':
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:  # Không còn detection nào để match
                break
            
            track_inds_l = [k for k in track_inds if tracks[k].time_since_update == 1 + level]
            
            if len(track_inds_l) == 0:          # Không còn track nào để match
                continue

            matches_l, _, unmatched_detections = min_cost_matching(distance_metric, match_thresh, tracks, detections,
                                                                    track_inds_l, unmatched_detections)
            
            matches += list(matches_l)
    
    unmatched_tracks = list(set(track_inds) - set(k for k, _ in matches))
    
    return matches, unmatched_tracks, unmatched_detections