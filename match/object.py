import numpy as np
from match.Kalman import KalmanFilter

class STrack():
    def __init__(self, position, feature, ID, state):
        self.step = 0
        self.states = ['new', 'tracking', 'unseen', 'lost']
        self.filter = KalmanFilter(position)
        self.position = position
        self.position_predict = position
        self.update_features(feature)
        self.alpha = 0.9
        self.lost = 0
        self.det = 0
        self.num_new2track = 3
        self.num_unseen2lost = 5
        self.num_unseen = 0
        self.ID = ID
        self.state = state
        self.position_yuanshi = position

    def update_features(self, feat):
        if self.step == 0:
            self.feature = feat
        if self.step>0 and self.step<10:
            cof = self.step*1.0/10
            self.feature = cof * self.feature + (1-cof)*feat
        if self.step > 0:
            cof = self.alpha
            self.feature = cof * self.feature + (1 - cof) * feat
        self.step = self.step + 1

    def update(self, position=None, feature=None):
        if position is not None:
            self.position, self.position_predict = self.filter.go(position)
            self.update_features(feature)
            self.det = self.det + 1
            self.position_yuanshi = position
        else:
            self.position, self.position_predict = self.filter.go()
            self.lost = self.lost + 1
            self.position_yuanshi = self.position
        self.state_check(position)

    def get_position(self):
        if self.state == 'new':
            return None
        if self.state == 'tracking':
            return self.position
        if self.state =='unseen':
            return self.position_predict
        if self.state == 'lost':
            return None

    def state_check(self, position):
        if position is None:
            self.state = 'unseen'
            self.lost = self.lost + 1
        else:
            if self.state == 'unseen':
                self.state = 'tracking'
                self.lost = 0

        if self.state == 'new':
            if self.det > self.num_new2track:
                self.state = 'tracking'
            else:
                self.det = self.det + 1
        if self.state == 'unseen':
            if self.lost > self.num_unseen2lost:
                self.state = 'lost'





