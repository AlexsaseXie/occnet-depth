import numpy as np
import torch

def adjust_learning_rate(optimizer, new_lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = new_lr

class StepLearningRateSchedule:
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)),1.0e-5)

def find_r_t(points):
    '''
        SAIL-S3: find r, t
        Input: points (n * 3) np array
    '''
    n = points.shape[0]
    A = np.concatenate([points * 2, np.ones((n,1))], axis=1) # n * 4
    y = (points * points).sum(axis=1)
    b = np.linalg.inv(A.T @ A) @ A.T @ y
    t = b[0:3]
    r = np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3])

    return r, t

def pcwrite(filename, xyzrgb, nxnynz=None, color=True, normal=False):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    if color:
        rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    if color:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
    if normal:
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
        ))

        if color:
            ply_file.write(" %d %d %d"%(
                rgb[i, 0], rgb[i, 1], rgb[i, 2],
            ))
        
        if normal:
            ply_file.write(" %f %f %f" % (
                nxnynz[i, 0], nxnynz[i, 1], nxnynz[i, 2]
            ))
        
        ply_file.write("\n")