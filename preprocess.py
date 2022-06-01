import torch

def reinforcement(tensor):
    h = tensor.size()[1]
    w = tensor.size()[2]

    for i in range(0, h):
        for j in range(0, w):
            if tensor[0][i][j] == tensor[0][0][0]:
                tensor[0][i][j] = 1.0
            if tensor[0][i][j] <= 0.999 and tensor[0][i][j] >= 0.0:
                tensor[0][i][j] = 0.0
            if tensor[0][i][j] <= 1.0 and tensor[0][i][j] > 0.999:
                tensor[0][i][j] = 1.0

    return tensor


def vampix(tensor):
    h = tensor.size()[1]
    w = tensor.size()[2]

    for i in range(0, h):
        for j in range(0, w):
            if tensor[0][i][j] == 0.0:
                tensor[0][i][j] = 1.0
            else:
                tensor[0][i][j] = 0.0

    return tensor

if __name__ == '__main__':
    dict1 = {'TP':{},
             'FP':{}}
    dict1['TP'][1] = 0
    print(dict1)