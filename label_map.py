import os


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


def create_label_map(signs, path):
    with open(path, "w"):
        pass
    for i in range(signs.shape[0]):
        label_sign = Label(i+1, signs[i])
        label_sign.make_text()
        label_sign.write_pbtxt(path)
    return path