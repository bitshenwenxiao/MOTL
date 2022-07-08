import os, cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from utils.reranking import re_ranking, re_ranking_numpy, batch_euclidean_distance

def compute_position_distance(position1, position2):
    position1 = np.array(position1)
    position2 = np.array(position2)
    vector_tem = position1 - position2
    dis = np.sqrt(np.inner(vector_tem, vector_tem))
    return dis

class generate_distance():
    def __init__(self, white):
        self.white = white
    def go(self, positions, features, targets):
        number_position = len(positions)
        number_target = len(targets)
        matrix_position = np.zeros((number_position, number_target))

        if number_position>0 and number_target>0:
            for pi in range(0, number_position):
                for ti in range(0, number_target):
                    matrix_position[pi,ti] = compute_position_distance(positions[pi], targets[ti].position_predict)

            features_qf = torch.zeros((max(number_position, 2), 2048)).cuda().float()
            features_gf = torch.zeros((max(number_target, 2), 2048)).cuda().float()
            for j in range(0, len(features)):
                features_qf[j] = features[j]
            for j, target in enumerate(targets):
                features_gf[j] = target.feature
            for j in range(number_position, 2):
                features_qf[j, :] = self.white
            for j in range(number_target, 2):
                features_gf[j, :] = self.white
            distmat0 = re_ranking(features_qf, features_gf, k1=2, k2=2, lambda_value=0.3)
            matrix_feature = distmat0[0:number_position, 0:number_target]
            matrix_cost = matrix_feature * matrix_position
            matrix_one = matrix_position.copy()
            matrix_one[np.where(matrix_one<0.3)] = 0.0
            matrix_cost = matrix_cost + matrix_one

        return matrix_cost

class image2feature():
    def __init__(self, model):
        self.val_transforms = T.Compose([
            T.Resize((384, 384), interpolation=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = model
    def go(self, image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        x = self.val_transforms(image).unsqueeze(0).float().cuda()
        feature = self.model(x)
        return feature[0, :].data



