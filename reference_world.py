import numpy as np

def ref3DModel():
    modelPoints = [[0.0, 0.0, 0.0],         #nose tip
                   [0.0, -330.0, -65.0],    #chin
                   [-225.0, 170.0, -135.0], #left eye corner
                   [225.0, 170.0, -135.0],  #right eye corner
                   [-150.0, -150.0, -125.0],#left mouth corner
                   [150.0, -150.0, -125.0]] #right mouth corner
    return np.array(modelPoints, dtype=np.float64)


def ref2dImagePoints(shape):
    imagePoints = [[shape.part(34).x, shape.part(34).y],
                   [shape.part(9).x, shape.part(9).y],
                   [shape.part(37).x, shape.part(37).y],
                   [shape.part(46).x, shape.part(46).y],
                   [shape.part(49).x, shape.part(49).y],
                   [shape.part(55).x, shape.part(55).y]]
    return np.array(imagePoints, dtype=np.float64)


def cameraMatrix(fl, heighttt, widthhh):
    mat = [[fl, 1, heighttt],
                    [0, fl, widthhh],
                    [0, 0, 1]]
    return np.array(mat, dtype=np.float64)
def cameraMatrix1(fl, center):
    mat = [[fl, 1, center[0]],
                    [0, fl, center[1]],
                    [0, 0, 1]]
    return np.array(mat, dtype=np.float64)