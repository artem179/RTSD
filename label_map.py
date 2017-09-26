import os
import numpy as np

class Label:
    def __init__(self, index, name):
        self.name = name
        self.index = index
    
    def make_text(self):
        self.text = "item {\n\tid: %d\n\tname: '%s'\n}\n" % (self.index, self.name)
    
    def write_pbtxt(self, path):
        with open(path, 'a') as file:
            file.write(self.text)
        file.close()


def create_label_map(signs, path, many_classes, group_of_class):
    with open(path, "w"):
        pass
    if many_classes:
        if group_of_class: signs = np.array([str(i) for i in range(1, 9)])
        for i in range(signs.shape[0]):
            label_sign = Label(i+1, signs[i])
            label_sign.make_text()
            label_sign.write_pbtxt(path)
    else:
        label_sign = Label(1, 'road_sign')
        label_sign.make_text()
        label_sign.write_pbtxt(path)
    return path