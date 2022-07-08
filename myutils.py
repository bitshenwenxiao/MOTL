import numpy as np
import math
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

def Euler2matrix(euler):
    angle_yaw, angle_pitch, angle_roll = euler * np.pi / 180.0
    psi, theta, gamma = angle_yaw, angle_pitch, angle_roll
    L_world2body = [[np.cos(psi) * np.cos(theta), np.sin(psi) * np.cos(theta), -np.sin(theta)],
                     [np.cos(psi) * np.sin(theta) * np.sin(gamma) - np.sin(psi) * np.cos(gamma),
                      np.sin(psi) * np.sin(theta) * np.sin(gamma) + np.cos(psi) * np.cos(gamma),
                      np.cos(theta) * np.sin(gamma)],
                     [np.cos(psi) * np.sin(theta) * np.cos(gamma) + np.sin(psi) * np.sin(gamma),
                      np.sin(psi) * np.sin(theta) * np.cos(gamma) - np.cos(psi) * np.sin(gamma),
                      np.cos(theta) * np.cos(gamma)]]
    return  np.array(L_world2body)
def matrix2Euler_nosign(L):
    i, j, k = 0, 1, 2
    yaw = math.atan(L[i, j] / L[i, i]) * 180 / np.pi
    pitch = math.atan2(-L[i, k], math.sqrt(L[i, j] * L[i, j] + L[i, i] * L[i, i])) * 180 / np.pi
    roll = math.atan(L[1, 2] / L[2, 2]) * 180 / np.pi
    return np.array([yaw, pitch, roll])
def matrix2Euler(L):
    angle = matrix2Euler_nosign(L)
    flag = False
    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                angle_tem = np.array([angle[0]+90*i, angle[1]+90*j, angle[2]+90*k])
                L2 = Euler2matrix(angle_tem)
                dif = np.sum(np.abs(L2-L))
                if dif < 0.001:
                    flag = True
                    return angle_tem
    if flag is False:
        return flag
def add_angle(angle_A,angle_B, angle_C):
    L_A = Euler2matrix(angle_A)
    L_B = Euler2matrix(angle_B)
    L_C = Euler2matrix(angle_C)
    L_tem = np.matmul(L_B, L_A)
    L = np.matmul(L_C, L_tem)
    angle_C = matrix2Euler(L)
    return angle_C
def Euler2vector(angle):
    L = Euler2matrix(angle)
    L2 = np.linalg.inv(L)
    x = np.array([1,0,0])
    x = x.reshape(3,1)
    y = np.matmul(L2, x)
    return y
