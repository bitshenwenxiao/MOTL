from utils.reranking import re_ranking
import torch, math
import numpy as np

def compute_perpendicular_foot_direct(point1, vector1, point2, vector2):
    a = np.inner(vector1, vector2)
    b = np.inner(vector1, vector1)
    c = np.inner(vector2, vector2)
    d = np.inner((point2-point1), vector1)
    e = np.inner((point2-point1), vector2)
    #if a == 0:
    if abs(a) < 0.01:
        t1 = d/b
        t2 = -e/c
        mod = 'perpendicular'
    elif ( a * a - b * c) == 0:
        t1 = 0
        t2 = -d/a
        mod = 'parallel'
    else:
        t1 = ( a * e - c * d) / ( a * a - b * c)
        t2 = b/a*t1-d/a
        mod = 'common'
    point1_tem = point1 + vector1 * t1
    point2_tem = point2 + vector2 * t2
    vector_tem = point2_tem - point1_tem
    dis = np.sqrt(np.inner(vector_tem, vector_tem))
    position = (point1_tem + point2_tem)/2
    if t1<0 or t2<0:
        dis = 100
    if mod == 'parallel':
        dis = 100
    cos = np.inner(vector1,vector2)/np.sqrt(np.inner(vector1,vector1))/np.sqrt(np.inner(vector2,vector2))
    return  dis, position

def compute_cost_geometry_spatial_onematch(query, candidate):
    m, n = len(query.bboxes), len(candidate.bboxes)
    out = np.zeros((m,n,4), np.float)
    for i in range(0, m):
        for j in range(0, n):
            point1, vector1 = query.vectors[i]
            point2, vector2 = candidate.vectors[j]
            out[i,j,0], out[i,j,1:4] = compute_perpendicular_foot_direct(point1, vector1, point2, vector2)
    return out[:,:,0], out[:,:,1:4]


def compute_cost_apperance_spatial_onematch(query, candidate, white):
    query_features = query.features
    candidate_features = candidate.features
    query_features_tem = torch.zeros((max(query_features.size(0), 2), 2048)).cuda().float()
    candidate_features_tem = torch.zeros((max(candidate_features.size(0), 2), 2048)).cuda().float()
    query_features_tem[0:query_features.size(0)] = query_features
    candidate_features_tem[0:candidate_features.size(0)] = candidate_features

    for j in range(query_features.size(0), 2):
        query_features_tem[j, :] = white
    for j in range(candidate_features.size(0), 2):
        query_features_tem[j, :] = white

    distmat = re_ranking(query_features_tem, candidate_features_tem, k1=2, k2=2, lambda_value=0.3)
    distmat = distmat[0:query_features.size(0), 0:candidate_features.size(0)]
    return distmat
