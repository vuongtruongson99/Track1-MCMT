import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack(object):
    _count = 0                  # Đếm số lượng track_id

    track_id = 0                
    is_activated = False
    state = TrackState.New

    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    # @property
    # def end_frame(self):
    #     return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0
