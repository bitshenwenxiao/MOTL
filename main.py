import os, cv2, torch
from config import cfg
import numpy as np
import argparse
from model import make_model
from  inputdata import dataset
from utils.reranking import re_ranking
from compute import compute_cost_apperance_spatial_onematch, compute_cost_geometry_spatial_onematch
from myutils import linear_assignment
from object import list_object
from match.track import TimeTrack
from match.distance import image2feature
class  Track():
    def __init__(self, model):
        self.model = model
        i2f = image2feature(model)
        white = np.zeros((384, 384, 3), np.uint8)
        self.white = i2f.go(white)
        self.time_tracker  = TimeTrack(self.white)
    def track(self, list_input):

        for input_data in list_input:
            input_data.get_features(self.model)
            input_data.get_vectors()
        cost_apperance_spatial = self.compute_cost_apperance_spatial(list_input)
        cost_geometry_spatial, positions = self.compute_cost_geometry_spatial(list_input)
        cost_spatial = self.fuse_cost(cost_apperance_spatial, cost_geometry_spatial)
        objects = self.spatial_assotation(cost_spatial, positions, list_input)
        positions = objects.get_positions()
        features = objects.get_features(list_input)
        self.time_tracker.track(positions, features)
        objects_with_ID = self.time_tracker.get_out()

        return objects_with_ID


    def fuse_cost(self, cost_A, cost_B):
        out = []
        for i in range(0, len(cost_A)):
            cost_apperance = cost_A[i]
            cost_geometry = cost_B[i]
            cost = cost_apperance * cost_geometry
            cost_one = cost_geometry.copy()
            cost_one[np.where(cost_one < 0.3)] = 0
            cost = cost + cost_one
            out.append(cost)
        return out
    def spatial_assotation(self, cost_spatial, positions, list_input):
        reports = []
        for i, cost in enumerate(cost_spatial):
            matches, unmatched_a, unmatched_b = linear_assignment(cost, 0.3)
            for ii, jj in matches:
                reports.append([1,self.UAVID[i][0], ii, self.UAVID[i][1], jj, positions[i][ii, jj, 0], positions[i][ii, jj, 1], positions[i][ii, jj, 2] ])
        objects = list_object()
        objects.add_reports(reports)
        objects.get_features(list_input)

        return objects
    def compute_cost_apperance_spatial(self, list_input):
        number = len(list_input)
        cost_apperance_spatial = []
        self.UAVID = []
        for i in range(0, number-1):
            for j in range(i + 1, number):
                self.UAVID.append([i,j])
                query = list_input[i]
                candidate = list_input[j]
                distmat = compute_cost_apperance_spatial_onematch(query, candidate,self.white)
                cost_apperance_spatial.append(distmat)

        return cost_apperance_spatial

    def compute_cost_geometry_spatial(self, list_input):
        number = len(list_input)
        cost_geometry_spatial = []
        positions = []
        for i in range(0, number-1):
            for j in range(i + 1, number):
                query = list_input[i]
                candidate = list_input[j]
                distmat, position = compute_cost_geometry_spatial_onematch(query, candidate)
                cost_geometry_spatial.append(distmat)
                positions.append(position)
        return cost_geometry_spatial, positions

def check_path(path):
    path_tem = path
    paths = path_tem.split('/')
    if path[0] == '/':
        path0 = '/' + paths[1]
        start = 2
    else:
        path0 = paths[0]
        start = 1
    for i in range(start, len(paths)):
        path0 = path0 + '/' + paths[i]
        if os.path.exists(path0) is False:
            os.mkdir(path0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="/home/station/code2/AICITY2021/configs/stage1/101a_384.yml",
        help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    model = make_model(cfg, num_class=1)
    model = model.cuda().eval()
    path_dataset = '/'
    path_det = 'result/detections/'
    path_out = './result/id/'
    for i in range(40,57):
        check_path(path_out + str(i))
        mydataset = dataset(str(i), path_dataset, path_det)
        mytrack = Track(model)

        j= 0
        while 1:
            j = j + 1
            print(i,j)
            list_input = mydataset.getdatas()
            if list_input is not False:
                objects_with_ID = mytrack.track(list_input)
                fout = open(path_out+str(i)+'/'+str(100000+j)+'.txt','w')
                for ID, position in objects_with_ID:
                    fout.write(str(ID) + ',' + str(position[0]) + ',' + str(position[1]) + ',' + str(position[2]) + '\n')
                fout.close()

            else:
                break

