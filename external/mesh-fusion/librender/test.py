import pyrender
import numpy as np
from matplotlib import pyplot
import math
import time

# render settings
img_h = 480
img_w = 480
fx = 480.
fy = 480.
cx = 240
cy = 240


def get_points(n_views):
    """
    See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.

    :param n_points: number of points
    :type n_points: int
    :return: list of points
    :rtype: numpy.ndarray
    """

    rnd = 1.
    points = []
    offset = 2. / n_views
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(n_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % n_views) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    # visualization.plot_point_cloud(np.array(points))
    return np.array(points)

def get_views(n_views):
    """
    Generate a set of views to generate depth maps from.

    :param n_views: number of views per axis
    :type n_views: int
    :return: rotation matrices
    :rtype: [numpy.ndarray]
    """

    Rs = []
    points = get_points(n_views)

    for i in range(points.shape[0]):
        # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
        longitude = - math.atan2(points[i, 0], points[i, 1])
        latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(latitude), -math.sin(latitude)],
                        [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array([[math.cos(longitude), 0, math.sin(longitude)],
                        [0, 1, 0],
                        [-math.sin(longitude), 0, math.cos(longitude)]])

        R = R_y.dot(R_x)
        Rs.append(R)

    return points, Rs

def model():

    # note that xx is height here!
    xx = -0.2
    yy = -0.2
    zz = -0.2

    v000 = (xx, yy, zz)  # 0
    v001 = (xx, yy, zz + 0.4)  # 1
    v010 = (xx, yy + 0.4, zz)  # 2
    v011 = (xx, yy + 0.4,  zz + 0.4)  # 3
    v100 = (xx + 0.4, yy, zz)  # 4
    v101 = (xx + 0.4, yy, zz + 0.4)  # 5
    v110 = (xx + 0.4, yy + 0.4, zz)  # 6
    v111 = (xx + 0.4, yy + 0.4, zz + 0.4)  # 7

    f1 = [0, 2, 4]
    f2 = [4, 2, 6]
    f3 = [1, 3, 5]
    f4 = [5, 3, 7]
    f5 = [0, 1, 2]
    f6 = [1, 3, 2]
    f7 = [4, 5, 7]
    f8 = [4, 7, 6]
    f9 = [4, 0, 1]
    f10 = [4, 5, 1]
    f11 = [2, 3, 6]
    f12 = [3, 7, 6]

    vertices = []
    vertices.append(v000)
    vertices.append(v001)
    vertices.append(v010)
    vertices.append(v011)
    vertices.append(v100)
    vertices.append(v101)
    vertices.append(v110)
    vertices.append(v111)

    faces = []
    faces.append(f1)
    faces.append(f2)
    faces.append(f3)
    faces.append(f4)
    faces.append(f5)
    faces.append(f6)
    faces.append(f7)
    faces.append(f8)
    faces.append(f9)
    faces.append(f10)
    faces.append(f11)
    faces.append(f12)

    return vertices, faces

def render(vertices, faces, R=None):
    if R is None:
        x = 0
        y = math.pi/4
        z = 0
        R_x = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        R_y = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        R_z = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        R = R_z.dot(R_y.dot(R_x))

    np_vertices = np.array(vertices).astype(np.float64)
    np_vertices = R.dot(np_vertices.T).T
    np_vertices[:, 2] += 1.0
    np_faces = np.array(faces).astype(np.float64)
    np_faces += 1

    depthmap, mask, img = pyrender.render(np_vertices.T.copy(), np_faces.T.copy(), np.array([fx, fy, cx, cy]), np.array([0.05, 2.]), np.array([img_h, img_w], dtype=np.int32))
    pyplot.imshow(depthmap)
    pyplot.show()
    pyplot.imshow(mask)
    pyplot.show()

def model_new():
    vertices, faces = model()

    vertices = np.array(vertices).astype(np.float32)
    faces = np.array(faces).astype(np.int)

    # no color info
    color_array = np.empty((0, 3), dtype=np.float32)

    vertex_array = vertices[faces].astype(np.float32) #F * 3 * 3
    normal_array = np.cross((vertex_array[:, 1, :] - vertex_array[:, 0, :]), (vertex_array[:, 2, :] - vertex_array[:, 0, :]))
    normal_norm = np.linalg.norm(normal_array, axis = 1).reshape(-1, 1)
    normal_array = normal_array / normal_norm
    normal_array = np.repeat(normal_array, 3, axis=0).copy()
    vertex_array = vertex_array.reshape(-1,3).copy()

    return vertex_array, normal_array, color_array

def render_new(vertex_array, normal_array, color_array, cam_position=None, show=True):
    if cam_position is None:
        x = 0
        y = math.pi/4
        z = 0

        cam_position = np.array([
            [1.5 * math.sin(y), 0, 1.5 * math.cos(y)],
            [1.5 * math.sin(y), 1.5 * math.cos(y), 0]
        ], dtype=np.float32)

    T = cam_position.shape[0]

    cam_intr = np.array([fx, fy, cx, cy], dtype=np.float32)
    znf = np.array([0.01, 2.], dtype=np.float32)
    img_size = np.array([img_h, img_w], dtype=np.int32)

    print('Testing rendering')
    t0 = time.time()
    depth, mask, img, normal, vertex, view_mats = pyrender.render_new(vertex_array, color_array, normal_array, cam_position, cam_intr, znf, img_size)
    print('render %d pics:' % cam_position.shape[0], time.time() - t0, 'secs')

    fusion_intrisics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    if show:
        for i in range(min(T, 5)):
            #pyplot.imshow(depth[i])
            #pyplot.show()

            normal_copy = normal.copy()
            normal_copy[i,:,:,3] /= 12
            normal_copy[i,:,:,:3] = (normal_copy[i,:,:,:3] + 1.) / 2.
            pyplot.imshow(normal_copy[i,:,:,:4])
            pyplot.show()

            vertex_copy = vertex.copy()
            vertex_copy[i,:,:,3] /= 12
            while True:
                x, y = np.random.randint(img_w), np.random.randint(img_h)
                vx = x
                vy = y
                v = vertex_copy[i, vy, vx, :3].copy()
                if v[0] != 0 or v[1] != 0 or v[2] != 0:
                    break

            print(x, y)
            print('v:', v)
            
            vertex_copy[i,:,:,:3] = (vertex_copy[i,:,:,:3] + 0.2) * (1.0 / 0.4)
            pyplot.imshow(vertex_copy[i,:,:,:4])
            
            # draw pixel
            view_mat = view_mats[i,:,:]
            view_mat[:,0] = -view_mat[:,0] # tranpose x axis
            v = np.append(v, [1.0], axis=0)

            t = np.dot(v, view_mat)[:3]
            t = np.dot(t, fusion_intrisics.T)
            t /= t[2] 
            print(t)
            pyplot.scatter(t[0], t[1], color='red')
            pyplot.show()

    print('Testing select')
    t0 = time.time()
    fM = vertex_array.shape[0] // 3
    pointcloud, stats = pyrender.select_vertex_from_buffer(normal, vertex, fM,  img_size, 5000000)
    print('select vertex:', time.time() - t0, 'secs')
    print('double-sides faces: %d, bad faces: %d, total faces: %d' % (stats[0], stats[1], stats[2]))
    print(pointcloud.shape)
    print(pointcloud)

if __name__ == '__main__':
    pts, Rs = get_views(100)

    #vertices, faces = model()
    #render(vertices, faces, R = Rs[30])

    vertex_array, normal_array, color_array = model_new()
    render_new(vertex_array, normal_array, color_array, cam_position = np.array(pts[:], dtype=np.float32), show=True)

    