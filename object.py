import numpy as np

class list_object():
    def __init__(self):
        self.list_object = []
    def get_features(self, list_input):
        features = []
        for target in self.list_object:
            feature = None
            for i,j in target.report_UAV:
                if feature is None:
                    feature = list_input[i].features[j]
                else:
                    feature = feature + list_input[i].features[j]
            feature = feature / len(target.report_UAV)
            features.append(feature)
        return  features
    def add_report(self, report):
        if report[0] < 0.5:
            self.add_report_dot(report)
        if report[0] > 0.5:
            self.add_report_edge(report)
    def add_reports(self, reports):
        for report in reports:
            self.add_report(report)
    def add_report_dot(self, report):
        id_save = self.search(report[1],report[2])
        if id_save > -1:
            self.list_object[id_save].add_reported(report)
        else:
            new_object = one_object()
            new_object.add_reported(report)
            self.list_object.append(new_object)
    def add_report_edge(self, report):
        id_save = self.search(report[1],report[2])
        if id_save > -1:
            self.list_object[id_save].add_reported(report)
        else:
            id_save = self.search(report[3], report[4])
            if id_save > -1:
                self.list_object[id_save].add_reported(report)
            else:
                new_object = one_object()
                new_object.add_reported(report)
                self.list_object.append(new_object)

    def search(self, ID_UAV, ID_object):
        flag = False
        for id_save, target in enumerate(self.list_object):
            for ID_UAV_tem, ID_object_tem in target.report_UAV:
                if ID_UAV_tem == ID_UAV and ID_object_tem==ID_object:
                    flag = True
                    return id_save
        if flag == False:
            return -1
    def get_positions(self):
        out = []
        for target in self.list_object:
            out.append(target.get_position())
        #out = np.array([out])
        self.positions = out
        return out

class one_object():
    def __init__(self):
        self.report_UAV = []
        self.positions = []
    def add_reported(self, report):
        self.report_UAV.append([report[1],report[2]])
        if report[0]>0.5:
            self.report_UAV.append([report[3], report[4]])
        self.positions.append(report)
    def get_distrance(self, position1, position2):
        cha = position1-position2
        dis = np.sqrt(np.inner(cha,cha))
        return  dis

    def get_position(self):
        position = np.zeros((3))
        count = 0
        for report in self.positions:
            if report[0] == 1:
                position = position + report[5:8]
                count = count + 1
        if count > 0:
            position = position/count
            return  position
if __name__ == '__main__':
    car = one_object()
    car.add_reported(np.array([1,1,1,2,1,2,2,2]))
    position = car.get_position()

    myobjects = list_object()
    myobjects.add_report([1,1,1,2,1,2,2,2])

    myobjects.add_report([1, 1, 1, 3, 1, 4, 2, 2])
    myobjects.add_report([1, 3, 1, 4, 1, 6, 2, 2])
    out = myobjects.get_positions()
    print(out)
