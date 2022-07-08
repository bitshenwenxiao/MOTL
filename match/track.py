from match.object import STrack
from match.distance import generate_distance
import numpy as np
import lap

def linear_assignment(cost_matrix, thresh):
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

class TimeTrack():
    def __init__(self, white):
        self.targets = []
        self.ID = 0
        self.GD = generate_distance(white)
    def initiate(self,positions, features):
        for i, position in enumerate(positions):
            self.targets.append(STrack(position, features[i], self.ID, 'tracking'))
            self.ID = self.ID + 1
    def track(self, positions, features):
        if len(self.targets) < 1 and len(positions) > 0:
            self.initiate(positions, features)

        if len(positions) < 1 and len(self.targets) > 0:
            for itrack in range(0, len(self.targets)):
                self.targets[itrack].update()
        if len(positions) > 0 and len(self.targets) > 0:
            cost = self.GD.go(positions, features, self.targets)
            matches, u_detections, u_tracks = linear_assignment(cost, thresh=0.2)
            for idet, itrack in matches:
                self.targets[itrack].update(positions[idet], features[idet])
            for idet in u_detections:
                self.targets.append(STrack(positions[idet], features[idet], self.ID, 'new'))
                self.ID = self.ID + 1
            for itrack in u_tracks:
                self.targets[itrack].update()
        targets_tem = []
        for target in self.targets:
            if target.state != 'lost':
                targets_tem.append(target)
            else:
                del target
        self.targets = targets_tem

    def get_out(self):
        out = []
        for target in self.targets:
            if target.state == 'tracking' or target.state == 'unseen':
                out.append([target.ID, target.position_yuanshi])
        return out


