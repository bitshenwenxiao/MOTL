import os,cv2,math
from myutils import add_angle, Euler2vector
import numpy as np
import torchvision.transforms as T
import torch
from PIL import Image
class Input():
    def __init__(self, image, position_UAV, Euler_inertial2body, Euler_body2camera, Euler_view, bboxes):
        self.image = image
        self.position_UAV = position_UAV
        self.Euler_inertial2body = Euler_inertial2body
        self.Euler_body2camera = Euler_body2camera
        self.Euler_view = Euler_view
        self.bboxes = bboxes
        self.val_transforms = T.Compose([
            T.Resize((384, 384), interpolation=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def get_features(self, model):

        out = []
        for i, bbox in enumerate(self.bboxes):
            patch = self.get_patch(bbox)
            x = self.val_transforms(patch).float()
            x = x.unsqueeze(0).cuda()
            out.append(model(x).data)
        self.features = torch.cat(out, dim=0)

        #self.patches = self.get_patches()
        #self.features = model(self.patches)
    def get_patches(self):
        out = torch.zeros((len(self.bboxes),3,384,384))
        for i, bbox in enumerate(self.bboxes):
            patch = self.get_patch(bbox)
            x = self.val_transforms(patch).float()
            out[i] = x[0,:].data
        out = out.cuda()
        self.patches = out
        return  out
    def get_patch(self,bbox):
        bbox = [int(item) for item in bbox]
        patch = self.image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        return  patch
    def get_vectors(self):
        out = []
        for i, bbox in enumerate(self.bboxes):
            data = bbox
            x, y = (data[0] + data[2]) / 2, (data[1] + data[3]) / 2
            angle_frame_0 = math.atan((x - 640.0) / 640.0 * np.tan(self.Euler_view[0] * np.pi / 180.0 / 2)) * 180 / np.pi
            angle_frame_1 = math.atan(-(y - 360.0) / 360.0 * np.tan(self.Euler_view[1] * np.pi / 180.0 / 2)) * 180 / np.pi
            angle_frame = np.array([angle_frame_0, angle_frame_1, 0])
            angle_out = add_angle(self.Euler_inertial2body, self.Euler_body2camera, angle_frame)
            vector = Euler2vector(angle_out)
            vector = vector.reshape(3)
            out.append([self.position_UAV, vector])
        self.vectors = out


class dataset():
    def __init__(self, seq, path_dataset, path_det):
        path_dataset = path_dataset + seq + '/'
        video_A = path_dataset + 'A/' + seq + '_A.avi'
        video_B = path_dataset + 'B/' + seq + '_B.avi'
        video_C = path_dataset + 'C/' + seq + '_C.avi'
        video_D = path_dataset + 'D/' + seq + '_D.avi'

        annos_A = path_dataset + 'A/' + seq + '_A.txt'
        annos_B = path_dataset + 'B/' + seq + '_B.txt'
        annos_C = path_dataset + 'C/' + seq + '_C.txt'
        annos_D = path_dataset + 'D/' + seq + '_D.txt'

        self.path_dets_A = path_det + seq + '_A/'
        self.path_dets_B = path_det + seq + '_B/'
        self.path_dets_C = path_det + seq + '_C/'
        self.path_dets_D = path_det + seq + '_D/'

        self.dets_A = os.listdir(self.path_dets_A)
        self.dets_B = os.listdir(self.path_dets_B)
        self.dets_C = os.listdir(self.path_dets_C)
        self.dets_D = os.listdir(self.path_dets_D)
        self.dets_A.sort()
        self.dets_B.sort()
        self.dets_C.sort()
        self.dets_D.sort()

        self.f_A = open(annos_A)
        self.f_B = open(annos_B)
        self.f_C = open(annos_C)
        self.f_D = open(annos_D)

        self.video_A = cv2.VideoCapture(video_A)
        self.video_B = cv2.VideoCapture(video_B)
        self.video_C = cv2.VideoCapture(video_C)
        self.video_D = cv2.VideoCapture(video_D)

        self.length = len(self.dets_A)
        self.iter = 0

    def getdatas(self):
        if self.iter < self.length:
            dataA = self.getdata(self.video_A, self.f_A, self.path_dets_A + self.dets_A[self.iter])
            dataB = self.getdata(self.video_B, self.f_B, self.path_dets_B + self.dets_B[self.iter])
            dataC = self.getdata(self.video_C, self.f_C, self.path_dets_C + self.dets_C[self.iter])
            dataD = self.getdata(self.video_D, self.f_D, self.path_dets_D + self.dets_D[self.iter])
            self.iter = self.iter + 1
            return [dataA, dataB, dataC, dataD]
        else:
            return  False
    def getdata(self, viedo, f, det):
        ret, image = viedo.read()
        data = f.readline()
        data = data.strip('\n').split(',')
        data = [float(item) for item in data]

        position_UAV = np.array([data[0], data[1], data[2]])
        Euler_inertial2body = np.array([data[3], data[4], data[5]])
        Euler_body2camera = np.array([data[6], data[7], data[8]])
        Euler_view = np.array([data[9], data[10]])

        f = open(det)
        datas = f.readlines()
        f.close()
        bboxes = []
        for data in datas:
            data = data.strip('\n').split(',')
            data = [float(item) for item in data]
            bboxes.append(data)
        out = Input(image, position_UAV, Euler_inertial2body, Euler_body2camera, Euler_view,bboxes)

        return  out



