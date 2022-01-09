import math
import numpy as np
import os
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator as rgi
import common
import argparse
import ntpath
import trimesh
import shutil

# Import shipped libraries.
import librender
import libmcubes
from multiprocessing import Pool

DEPTH_VIS_OUTPUT = False
POINTCLOUD_VIS_OUTPUT = False
use_gpu = True
if use_gpu:
    import libfusiongpu as libfusion
    from libfusiongpu import tsdf_gpu as compute_tsdf
    from libfusiongpu import tsdf_strict_gpu as compute_tsdf_strict
    from libfusiongpu import tsdf_range_gpu as compute_tsdf_range
    from libfusiongpu import judge_inside as compute_judge_inside
    from libfusiongpu import view_pc_tsdf_estimation as compute_view_pc_tsdf
    from libfusiongpu import view_pc_tsdf_estimation_var as compute_view_pc_tsdf_var
else:
    import libfusioncpu as libfusion
    from libfusioncpu import tsdf_cpu as compute_tsdf


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

class Fusion:
    """
    Performs TSDF fusion.
    """

    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()

        self.render_intrinsics = np.array([
            self.options.focal_length_x,
            self.options.focal_length_y,
            self.options.principal_point_x,
            self.options.principal_point_x
        ], dtype=float)
        # Essentially the same as above, just a slightly different format.
        self.fusion_intrisics = np.array([
            [self.options.focal_length_x, 0, self.options.principal_point_x],
            [0, self.options.focal_length_y, self.options.principal_point_y],
            [0, 0, 1]
        ])
        self.image_size = np.array([
            self.options.image_height,
            self.options.image_width,
        ], dtype=np.int32)
        # Mesh will be centered at (0, 0, 1)!
        # self.znf = np.array([
        #     1 - 0.75,
        #     1 + 0.75
        # ], dtype=float)
        self.znf = np.array([
            0.05,
            10
        ], dtype=float)

        # Derive voxel size from resolution.
        self.voxel_size = 1./self.options.resolution
        self.truncation = self.options.truncation_factor*self.voxel_size

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--mode', type=str, default='render',
                            help='Operation mode: render, fuse or sample.')
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument('--in_dir', type=str,
                                 help='Path to input directory.')
        input_group.add_argument('--in_file', type=str,
                                 help='Path to input directory.')
        parser.add_argument('--out_dir', type=str,
                            help='Path to output directory; files within are overwritten!')
        parser.add_argument('--t_dir', type=str,
                            help='Path to transformation directory.')
        parser.add_argument('--n_proc', type=int, default=0,
                            help='Number of processes to run in parallel'
                                 '(0 means sequential execution).')
        parser.add_argument('--overwrite', action='store_true',
                            help='Overwrites existing files if true.')

        parser.add_argument('--n_points', type=int, default=100000,
                            help='Number of points to sample per model.')
        parser.add_argument('--n_views', type=int, default=100,
                            help='Number of views per model.')
        parser.add_argument('--image_height', type=int, default=640,
                            help='Depth image height.')
        parser.add_argument('--image_width', type=int, default=640,
                            help='Depth image width.')
        parser.add_argument('--focal_length_x', type=float, default=640,
                            help='Focal length in x direction.')
        parser.add_argument('--focal_length_y', type=float, default=640,
                            help='Focal length in y direction.')
        parser.add_argument('--principal_point_x', type=float, default=320,
                            help='Principal point location in x direction.')
        parser.add_argument('--principal_point_y', type=float, default=320,
                            help='Principal point location in y direction.')
        parser.add_argument('--sample_weighted', action='store_true',
                            help='Whether to use weighted sampling.')
        parser.add_argument('--sample_scale', type=float, default=0.2,
                            help='Scale for weighted sampling.')
        parser.add_argument(
            '--depth_offset_factor', type=float, default=1.5,
            help='The depth maps are offsetted using depth_offset_factor*voxel_size.')
        parser.add_argument('--resolution', type=float, default=256,
                            help='Resolution for fusion.')
        parser.add_argument(
            '--truncation_factor', type=float, default=10,
            help='Truncation for fusion is derived as truncation_factor*voxel_size.')

        parser.add_argument('--type', type=str, default='tsdf', help='tsdf fusion type')

        # sample related params
        parser.add_argument('--points_size', type=int, default=100000)
        parser.add_argument('--points_uniform_ratio', type=float, default=1.)
        parser.add_argument('--points_padding', type=float, default=0.1)
        parser.add_argument('--outside_min_view', type=int, default=1)
        parser.add_argument('--bbox_in_folder', type=str, default=None)
        parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
        parser.add_argument('--float16', action='store_true', help='Whether to use half precision.')
        parser.add_argument('--packbits', action='store_true', help='Whether to save truth values as bit array.')
        parser.add_argument('--pointcloud_folder', type=str, default='./4_pointcloud/')
        parser.add_argument('--pointcloud_size', type=int, default=100000, help='Size of point cloud.')
        parser.add_argument('--points_according_to_pc_uniform_size', type=int, default=20000)
        parser.add_argument('--points_according_to_pc_size', type=int, default=100000)
        parser.add_argument('--points_pc_box', type=float, default=0.1)
        parser.add_argument('--tsdf_offset', type=float, default=0.008, help='offset for tsdf to decide the surface')
        return parser

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            if filename.endswith('.npz') or filename.endswith('.off') or filename.endswith('.h5') or filename.endswith('.ply'):
                files.append(os.path.normpath(os.path.join(directory, filename)))

        return files

    def get_in_files(self):
        if self.options.in_dir is not None:
            assert os.path.exists(self.options.in_dir)
            common.makedir(self.options.out_dir)
            files = self.read_directory(self.options.in_dir)
        else:
            files = [self.options.in_file]

        if not self.options.overwrite:
            def file_filter(filepath):
                outpath = self.get_outpath(filepath)
                return not os.path.exists(outpath)
            files = list(filter(file_filter, files))

        return files

    def get_outpath(self, filepath):
        filename = os.path.basename(filepath)
        if self.options.mode in ('render', 'render_new'):
            outpath = os.path.join(self.options.out_dir, filename + '.h5')
        elif self.options.mode == 'fuse':
            modelname = os.path.splitext(os.path.splitext(filename)[0])[0]
            outpath = os.path.join(self.options.out_dir, modelname + '.off')
        elif self.options.mode in ('judge_inside_simple', 'judge_tsdf_view_pc', 'judge_tsdf_view_pc_according_to_pc'):
            modelname = os.path.splitext(os.path.splitext(filename)[0])[0]
            outpath = os.path.join(self.options.out_dir, modelname + '.npz')
        elif self.options.mode == 'sample':
            modelname = os.path.splitext(os.path.splitext(filename)[0])[0]
            outpath = os.path.join(self.options.out_dir, modelname + '.npz')

        return outpath
        
    def get_points(self):
        """
        See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.

        :param n_points: number of points
        :type n_points: int
        :return: list of points
        :rtype: numpy.ndarray
        """

        rnd = 1.
        points = []
        offset = 2. / self.options.n_views
        increment = math.pi * (3. - math.sqrt(5.))

        for i in range(self.options.n_views):
            y = ((i * offset) - 1) + (offset / 2)
            r = math.sqrt(1 - pow(y, 2))

            phi = ((i + rnd) % self.options.n_views) * increment

            x = math.cos(phi) * r
            z = math.sin(phi) * r

            points.append([x, y, z])

        # visualization.plot_point_cloud(np.array(points))
        return np.array(points)

    def get_views(self):
        """
        Generate a set of views to generate depth maps from.

        :param n_views: number of views per axis
        :type n_views: int
        :return: rotation matrices
        :rtype: [numpy.ndarray]
        """

        Rs = []
        points = self.get_points()

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

        return Rs

    def render(self, mesh, Rs):
        """
        Render the given mesh using the generated views.

        :param base_mesh: mesh to render
        :type base_mesh: mesh.Mesh
        :param Rs: rotation matrices
        :type Rs: [numpy.ndarray]
        :return: depth maps
        :rtype: numpy.ndarray
        """

        depthmaps = []
        for i in range(len(Rs)):
            np_vertices = Rs[i].dot(mesh.vertices.astype(np.float64).T)
            np_vertices[2, :] += 1

            np_faces = mesh.faces.astype(np.float64)
            np_faces += 1

            depthmap, mask, img = \
                librender.render(np_vertices.copy(), np_faces.T.copy(),
                                 self.render_intrinsics, self.znf, self.image_size)

            # This is mainly result of experimenting.
            # The core idea is that the volume of the object is enlarged slightly
            # (by subtracting a constant from the depth map).
            # Dilation additionally enlarges thin structures (e.g. for chairs).
            depthmap -= self.options.depth_offset_factor * self.voxel_size
            #depthmap = ndimage.morphology.grey_erosion(depthmap, size=(3, 3))

            depthmaps.append(depthmap)

        return depthmaps

    def render_new(self, mesh, camera_positions):
        np_vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces
        np_normals = mesh.face_normals.astype(np.float32)

        # process: remove area == 0 triangles
        np_normals_len = np.linalg.norm(np_normals, axis=1, keepdims=False)
        good_id = np_normals_len > 0.5
        np_normals = np_normals[good_id, :]
        faces = faces[good_id, :]

        F = faces.shape[0]
        T = camera_positions.shape[0]

        render_np_vertices = np_vertices[faces].reshape(F * 3, 3)
        render_np_colors = np.zeros((0,3), dtype=np.float32)
        render_np_normals = np_normals.repeat(3, axis=0) # (F*3) * 3

        if POINTCLOUD_VIS_OUTPUT:
            print('Before rendering:')
            print('Normal nan: %d' % np.isnan(render_np_normals).sum())
            normal_lens = np.linalg.norm(render_np_normals, axis=1, keepdims=False)
            bad_ids = normal_lens < 0.5
            if bad_ids.sum() > 0:
                print('Bad verts:')
                print(render_np_vertices[bad_ids, :])
            print('Vertex nan: %d' % np.isnan(render_np_vertices).sum())

        #print("vert:", render_np_vertices.shape, "colors:", render_np_colors.shape, "normals:", render_np_vertices.shape)
        #print("vert:", render_np_vertices[0:6,:])
        #print("normals:", render_np_normals[0:6,:])
        #print('Begin render_new')
        depth, mask, img, normal, vertex, view_mat = librender.render_new(render_np_vertices, render_np_colors, render_np_normals, 
                                (camera_positions).astype(np.float32), 
                                self.render_intrinsics.astype(np.float32),
                                self.znf.astype(np.float32), self.image_size)


        if DEPTH_VIS_OUTPUT:
            from matplotlib import pyplot

            i = 5
            vertex_copy = vertex.copy()
            vertex_copy[i,:,:,3] = (vertex_copy[i,:,:,3]) / F
            pyplot.imshow(vertex_copy[i,:,:,:4])
            pyplot.show()


        return depth, mask, img, normal, vertex, view_mat

    def select_vertex_new(self, mesh, normal, vertex, sample_strategy='random', sample_n=100000):
        faces = mesh.faces
        F = faces.shape[0]

        assert sample_strategy in (None, 'random', 'fps')

        if sample_strategy in (None, 'random'):
            pointcloud, face_normal, stats = librender.select_vertex_from_buffer(normal, vertex, F, self.image_size, sample_n)

            # pc_size = pointcloud.shape[0]
            # if (pc_size >= sample_n):
            #     idx = np.random.choice(range(pc_size), size=sample_n, replace=False)
            # else:
            #     all_idx = np.arange(pc_size, dtype=np.int)
            #     random_idx = np.random.randint(pc_size, size=sample_n - pc_size, dtype=np.int)
            #     idx = np.concatenate((all_idx, random_idx), axis=0)
            # pointcloud = pointcloud[idx]
        elif sample_strategy == 'fps':
            #TODO
            pointcloud, face_normal, stats = librender.select_vertex_from_buffer(normal, vertex, F, self.image_size, sample_n * 10)

            pass
        else:
            raise NotImplementedError

        assert pointcloud.shape[0] == sample_n
        return pointcloud, face_normal, stats

    def fusion(self, depthmaps, Rs):
        """
        Fuse the rendered depth maps.

        :param depthmaps: depth maps
        :type depthmaps: numpy.ndarray
        :param Rs: rotation matrices corresponding to views
        :type Rs: [numpy.ndarray]
        :return: (T)SDF
        :rtype: numpy.ndarray
        """

        Ks = self.fusion_intrisics.reshape((1, 3, 3))
        Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

        Ts = []
        for i in range(len(Rs)):
            Rs[i] = Rs[i]
            Ts.append(np.array([0, 0, 1]))

        Ts = np.array(Ts).astype(np.float32)
        Rs = np.array(Rs).astype(np.float32)

        depthmaps = np.array(depthmaps).astype(np.float32)
        views = libfusion.PyViews(depthmaps, Ks, Rs, Ts)

        # Note that this is an alias defined as libfusiongpu.tsdf_gpu or libfusioncpu.tsdf_cpu!
        if self.options.type == 'tsdf':
            tsdf = compute_tsdf(views,
                            self.options.resolution, self.options.resolution,
                            self.options.resolution, self.voxel_size, self.truncation, False)
        elif self.options.type == 'tsdf_strict':
            tsdf = compute_tsdf_strict(views,
                            self.options.resolution, self.options.resolution,
                            self.options.resolution, self.voxel_size, self.truncation, False)
        elif self.options.type == 'tsdf_range':
            tsdf = compute_tsdf_range(views,
                            self.options.resolution, self.options.resolution,
                            self.options.resolution, self.voxel_size, self.truncation, True)
        else:
            raise NotImplementedError

        tsdf = np.transpose(tsdf[0], [2, 1, 0])
        return tsdf

    def judge_inside_simple(self, depthmaps, Rs, points_buffer):
        Ks = self.fusion_intrisics.reshape((1, 3, 3))
        Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

        Ts = []
        for i in range(len(Rs)):
            Rs[i] = Rs[i]
            Ts.append(np.array([0, 0, 1]))

        Ts = np.array(Ts).astype(np.float32)
        Rs = np.array(Rs).astype(np.float32)

        depthmaps = np.array(depthmaps).astype(np.float32)
        views = libfusion.PyViews(depthmaps, Ks, Rs, Ts)

        return compute_judge_inside(views, points_buffer)

    def judge_tsdf_view_pc(self, depthmaps, Rts, pointcloud, points, truncation=10./256., aggregate='min'):
        Ks = self.fusion_intrisics.reshape((1, 3, 3))
        Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

        #print('Rts.shape:', Rts.shape)
        Rts = np.array(Rts).astype(np.float32).transpose((0,2,1))
        # flip y z axes
        Rts[:,1:3,:] = -Rts[:,1:3,:]
        Ts = np.ascontiguousarray(Rts[:,:3,3])
        #print('Ts:',Ts)
        Rs = np.ascontiguousarray(Rts[:,:3,:3])
        #print('Rs:', Rs)

        depthmaps = np.array(depthmaps).astype(np.float32)
        views = libfusion.PyViews(depthmaps, Ks, Rs, Ts)

        #tsdf = compute_view_pc_tsdf(views, pointcloud, points, truncation, aggregate)
        tsdf = compute_view_pc_tsdf_var(views, pointcloud, points, truncation, aggregate)
        return tsdf

    def run(self):
        """
        Run the tool.
        """
        common.makedir(self.options.out_dir)
        files = self.get_in_files()

        if self.options.mode == 'render':
            method = self.run_render
        elif self.options.mode == 'fuse':
            method = self.run_fuse
        elif self.options.mode == 'sample':
            method = self.run_sample
        elif self.options.mode == 'judge_inside_simple':
            method = self.run_judge_inside_simple
        elif self.options.mode == 'render_new':
            method = self.run_render_new
        elif self.options.mode == 'judge_tsdf_view_pc':
            method = self.run_tsdf_view_pc
        elif self.options.mode == 'judge_tsdf_view_pc_according_to_pc':
            method = self.run_tsdf_view_pc_according_to_pc
        else:
            print('Invalid model, choose render or fuse.')
            exit()

        if self.options.n_proc == 0:
            for filepath in files:
                method(filepath)
        else:
            with Pool(self.options.n_proc) as p:
                p.map(method, files)

    def run_render(self, filepath):
        """
        Run rendering.
        """
        timer = common.Timer()
        Rs = self.get_views()

        timer.reset()
        mesh = common.Mesh.from_off(filepath)
        depths = self.render(mesh, Rs)

        depth_file = self.get_outpath(filepath)
        common.write_hdf5(depth_file, np.array(depths))
        print('[Data] wrote %s (%f seconds)' % (depth_file, timer.elapsed()))

    def run_render_new(self, filepath):
        """
        Run rendering.
        """
        timer = common.Timer()
        #Rs = self.get_views()
        cam_positions = self.get_points()

        timer.reset()
        mesh = trimesh.load(filepath)
        depth, mask, img, normal, vertex, view_mat  = self.render_new(mesh, cam_positions)
        modelname = os.path.splitext(os.path.splitext(os.path.basename(filepath))[0])[0]

        pointcloud, face_normal, stats = self.select_vertex_new(mesh, normal, vertex, sample_strategy='random', sample_n=self.options.pointcloud_size)

        double_sided_face_count = stats[0]
        bad_face_count = stats[1]
        total_visible_face_count = stats[2]


        print('Double side: %d, Bad: %d, Visible: %d' % (double_sided_face_count, bad_face_count, total_visible_face_count))
        if (double_sided_face_count >= total_visible_face_count * 0.1):
            print('%s Too much double sided faces' % modelname)
            with open(os.path.join(self.options.out_dir, 'removed_list.txt'), 'a') as f:    
                f.write('%s\n' % modelname)

            removed_copy_path = os.path.join(self.options.out_dir, 'removed')
            common.makedir(removed_copy_path)
            shutil.copy(filepath, removed_copy_path)
            return
        
        if (bad_face_count >= total_visible_face_count * 0.1):
            print('%s Too much bad faces' % modelname)
            with open(os.path.join(self.options.out_dir, 'removed_list.txt'), 'a') as f:    
                f.write('%s\n' % modelname)

            removed_copy_path = os.path.join(self.options.out_dir, 'removed')
            common.makedir(removed_copy_path)
            shutil.copy(filepath, removed_copy_path)
            return

        data_dict = {
            'depth': depth,
            #'normal': normal,
            #'vertex': vertex,
            #'face_normal': face_normal,
            'view_mat': view_mat,
            'stats': stats
        }

        if POINTCLOUD_VIS_OUTPUT:
            print('Normal nan: %d' % np.isnan(normal).sum())
            print('Vertex nan: %d' % np.isnan(vertex).sum())
            print('Face normal nan: %d' % np.isnan(face_normal).sum())

        depth_file = self.get_outpath(filepath)
        common.write_hdf5_dict(depth_file, data_dict)
        print('[Data] wrote %s (%f seconds)' % (depth_file, timer.elapsed()))

        modelname = os.path.splitext(os.path.splitext(os.path.basename(filepath))[0])[0]
        loc, scale, padding = self.get_transform(modelname)

        scale = scale * (1 - padding) / (1 - self.options.bbox_padding)
        filename = os.path.join(self.options.pointcloud_folder, '%s.npz' % modelname)

        points_scale = (1 - self.options.bbox_padding) / (1 - padding)
        points = pointcloud[:,:3] * points_scale
        normals = pointcloud[:,3:6]
        np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)

        if POINTCLOUD_VIS_OUTPUT:
            filename = os.path.join(self.options.pointcloud_folder, '%s.ply' % modelname)
            pcwrite(filename, points / points_scale, nxnynz=normals, color=False, normal=True)

    def run_fuse(self, filepath):
        """
        Run fusion.
        """
        timer = common.Timer()
        Rs = self.get_views()

        # As rendering might be slower, we wait for rendering to finish.
        # This allows to run rendering and fusing in parallel (more or less).
        depths = common.read_hdf5(filepath)

        timer.reset()
        tsdf = self.fusion(depths, Rs)
        # To ensure that the final mesh is indeed watertight
        tsdf = np.pad(tsdf, 1, 'constant', constant_values=1e6)
        vertices, triangles = libmcubes.marching_cubes(-tsdf, 0)
        # Remove padding offset
        vertices -= 1
        # Normalize to [-0.5, 0.5]^3 cube
        vertices /= self.options.resolution
        vertices -= 0.5

        modelname = os.path.splitext(os.path.splitext(os.path.basename(filepath))[0])[0]
        t_loc, t_scale, _ = self.get_transform(modelname)
        vertices = t_loc + t_scale * vertices

        off_file = self.get_outpath(filepath)
        libmcubes.export_off(vertices, triangles, off_file)
        print('[Data] wrote %s (%f seconds)' % (off_file, timer.elapsed()))

    def run_judge_inside_simple(self, filepath):
        timer = common.Timer()
        Rs = self.get_views()

        # As rendering might be slower, we wait for rendering to finish.
        # This allows to run rendering and fusing in parallel (more or less).
        depths = common.read_hdf5(filepath)
        timer.reset()

        modelname = os.path.splitext(os.path.splitext(os.path.basename(filepath))[0])[0]
        loc, scale, padding = self.get_transform(modelname)

        scale = scale * (1 - padding) / (1 - self.options.bbox_padding)

        n_points_uniform = int(self.options.points_size * self.options.points_uniform_ratio)
        assert self.options.points_uniform_ratio == 1.

        boxsize = 1 + self.options.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3) # dtype == np.float64
        points_uniform = boxsize * (points_uniform - 0.5)

        points = points_uniform.astype(np.float32)
        # normalize
        points_buffer = points * (1 - padding) / (1 - self.options.bbox_padding)
        points_buffer = self.judge_inside_simple(depths, Rs, points_buffer)

        points_outside_view_count = points_buffer[:, 3]
        print('Points_outside_view_count max:', points_outside_view_count.max(), 
            'min:', points_outside_view_count.min())
        #print(points_outside_view_count)
        points_inside_view_count = points_buffer[:, 4]
        #print(points_inside_view_count)
        #print(points_inside_view_count + points_outside_view_count)
        #points_inside_view_count = np.floor(points_buffer[:, 5] + 0.1).astype(np.int)

        occupancies = ~(points_outside_view_count > self.options.outside_min_view - 0.5)
        print('Volume: %d/%d' % (occupancies.sum(), occupancies.shape[0]))

        off_file = self.get_outpath(filepath)
        if True:
            inside_points = points[occupancies == 1]
            ply_path = off_file + ".ply"
            pcwrite(ply_path, inside_points, color=False)

        if self.options.float16:
            points = points.astype(np.float16)
        if self.options.packbits:
            occupancies = np.packbits(occupancies)

        np.savez(off_file, points=points, occupancies=occupancies,
             loc=loc, scale=scale)
        print('[Data] wrote %s (%f seconds)' % (off_file, timer.elapsed()))

    def run_tsdf_view_pc(self, filepath):
        timer = common.Timer()

        # As rendering might be slower, we wait for rendering to finish.
        # This allows to run rendering and fusing in parallel (more or less).
        data_dict = common.read_hdf5_dict(filepath)
        depths = data_dict['depth']
        # Rs = self.get_views()
        Rts = data_dict['view_mat']
        
        #stats = data_dict['stats']
        timer.reset()

        modelname = os.path.splitext(os.path.splitext(os.path.basename(filepath))[0])[0]
        loc, scale, padding = self.get_transform(modelname)

        scale = scale * (1 - padding) / (1 - self.options.bbox_padding)

        filename = os.path.join(self.options.pointcloud_folder, '%s.npz' % modelname)
        pointcloud_npz = np.load(filename)

        pointcloud_scale = (1 - padding) / (1 - self.options.bbox_padding)
        pointcloud_points = pointcloud_npz['points'] * pointcloud_scale
        pointcloud_normal = pointcloud_npz['normals']
        pointcloud = np.concatenate((pointcloud_points, pointcloud_normal), axis=1)

        n_points_uniform = int(self.options.points_size * self.options.points_uniform_ratio)
        assert self.options.points_uniform_ratio == 1.

        boxsize = 1 + self.options.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3) # dtype == np.float64
        points_uniform = boxsize * (points_uniform - 0.5)

        points = points_uniform.astype(np.float32)
        # normalize
        points_buffer = points * (1 - padding) / (1 - self.options.bbox_padding)
        #tsdf = self.judge_tsdf_view_pc(depths, Rts, pointcloud, points_buffer, truncation=self.truncation, aggregate='mean')
        tsdf = self.judge_tsdf_view_pc(depths, Rts, pointcloud, points_buffer, truncation=self.truncation, aggregate='min')

        occupancies = tsdf < self.options.tsdf_offset
        print('Volume: %d/%d' % (occupancies.sum(), occupancies.shape[0]))

        off_file = self.get_outpath(filepath)
        if POINTCLOUD_VIS_OUTPUT:
            inside_points = points_buffer[occupancies == 1]
            ply_path = off_file + ".ply"
            pcwrite(ply_path, inside_points, color=False)

            outside_points = points_buffer[occupancies == 0]
            ply_path = off_file + "_out.ply"
            pcwrite(ply_path, outside_points, color=False)

        if self.options.float16:
            points = points.astype(np.float16)
        if self.options.packbits:
            occupancies = np.packbits(occupancies)

        np.savez(off_file, points=points, occupancies=occupancies, tsdf=tsdf,
             loc=loc, scale=scale)
        print('[Data] wrote %s (%f seconds)' % (off_file, timer.elapsed()))

    def run_tsdf_view_pc_according_to_pc(self, filepath):
        timer = common.Timer()

        # As rendering might be slower, we wait for rendering to finish.
        # This allows to run rendering and fusing in parallel (more or less).
        data_dict = common.read_hdf5_dict(filepath)
        depths = data_dict['depth']
        # Rs = self.get_views()
        Rts = data_dict['view_mat']
        
        #stats = data_dict['stats']
        timer.reset()

        modelname = os.path.splitext(os.path.splitext(os.path.basename(filepath))[0])[0]
        loc, scale, padding = self.get_transform(modelname)

        scale = scale * (1 - padding) / (1 - self.options.bbox_padding)

        filename = os.path.join(self.options.pointcloud_folder, '%s.npz' % modelname)
        pointcloud_npz = np.load(filename)
        pointcloud_normalized = pointcloud_npz['points']

        pointcloud_scale = (1 - padding) / (1 - self.options.bbox_padding)
        pointcloud_points = pointcloud_normalized * pointcloud_scale
        pointcloud_normal = pointcloud_npz['normals']
        pointcloud = np.concatenate((pointcloud_points, pointcloud_normal), axis=1)

        n_points_uniform = self.options.points_according_to_pc_uniform_size
        n_points_according_to_pc = self.options.points_according_to_pc_size

        # sample code
        # uniform
        boxsize = 1 + self.options.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3) # dtype == np.float64
        points_uniform = (boxsize * (points_uniform - 0.5)).astype(np.float32)

        # according to pc
        pointcloud_size = pointcloud_normalized.shape[0]
        idx = np.random.randint(pointcloud_size, size=n_points_according_to_pc)
        points_pc = pointcloud_normalized[idx,:].astype(np.float32)
        displacement = np.random.rand(n_points_according_to_pc, 3)
        displacement = (self.options.points_pc_box * (displacement - 0.5)).astype(np.float32)
        points_pc = points_pc + displacement
        
        points = np.concatenate((points_uniform, points_pc), axis=0)
        # normalize
        points_buffer = points * (1 - padding) / (1 - self.options.bbox_padding)
        #tsdf = self.judge_tsdf_view_pc(depths, Rts, pointcloud, points_buffer, truncation=self.truncation, aggregate='mean')
        tsdf = self.judge_tsdf_view_pc(depths, Rts, pointcloud, points_buffer, truncation=self.truncation, aggregate='min')
        occupancies = tsdf < self.options.tsdf_offset

        off_file = self.get_outpath(filepath)
        if POINTCLOUD_VIS_OUTPUT:
            a_occupancies = occupancies[:n_points_uniform]
            a_points_buffer = points_buffer[:n_points_uniform]

            a_color = np.ones((n_points_uniform, 3), dtype=np.float32)
            a_color[a_occupancies == 1, :] = np.array([255,255,255], dtype=np.float32)

            a_xyzrgb = np.concatenate((a_points_buffer, a_color), axis=1)
            print(a_xyzrgb.shape)
            ply_path = off_file + "uniform.ply"
            pcwrite(ply_path, a_xyzrgb, color=True)

            a_occupancies = occupancies[n_points_uniform:]
            a_points_buffer = points_buffer[n_points_uniform:]

            inside_points = a_points_buffer[a_occupancies == 1]
            ply_path = off_file + ".ply"
            pcwrite(ply_path, inside_points, color=False)

            outside_points = a_points_buffer[a_occupancies == 0]
            ply_path = off_file + "_out.ply"
            pcwrite(ply_path, outside_points, color=False)

        if self.options.float16:
            points = points.astype(np.float16)
        if self.options.packbits:
            occupancies = np.packbits(occupancies)

        np.savez(off_file, points=points, occupancies=occupancies, tsdf=tsdf,
             loc=loc, scale=scale)
        print('[Data] wrote %s (%f seconds)' % (off_file, timer.elapsed()))

    def run_sample(self, filepath):
        """
        Run sampling.
        """
        timer = common.Timer()
        Rs = self.get_views()

        # As rendering might be slower, we wait for rendering to finish.
        # This allows to run rendering and fusing in parallel (more or less).

        depths = common.read_hdf5(filepath)

        timer.reset()
        tsdf = self.fusion(depths, Rs)

        xs = np.linspace(-0.5, 0.5, tsdf.shape[0])
        ys = np.linspace(-0.5, 0.5, tsdf.shape[1])
        zs = np.linspace(-0.5, 0.5, tsdf.shape[2])
        tsdf_func = rgi((xs, ys, zs), tsdf)

        modelname = os.path.splitext(os.path.splitext(os.path.basename(filepath))[0])[0]
        points = self.get_random_points(tsdf)
        values = tsdf_func(points)
        t_loc, t_scale, _ = self.get_transform(modelname)

        occupancy = (values <= 0.)
        out_file = self.get_outpath(filepath)
        np.savez(out_file, points=points, occupancy=occupancy, loc=t_loc, scale=t_scale)

        print('[Data] wrote %s (%f seconds)' % (out_file, timer.elapsed()))

    def get_transform(self, modelname):
        if self.options.t_dir is not None:
            t_filename = os.path.join(self.options.t_dir, modelname + '.npz')
            t_dict = np.load(t_filename)
            t_loc = t_dict['loc']
            t_scale = t_dict['scale']
            if 'padding' in t_dict:
                t_padding = t_dict['padding']
            else:
                t_padding = 0.1
        else:
            t_loc = np.zeros(3)
            t_scale = np.ones(3)
            t_padding = 0.1

        return t_loc, t_scale, t_padding

    def get_random_points(self, tsdf):
        N1, N2, N3 = tsdf.shape
        npoints = self.options.n_points

        if not self.options.sample_weighted:
            points = np.random.rand(npoints, 3)
        else:
            df = np.abs(tsdf)
            scale = self.options.sample_scale * df.max()
            indices = np.arange(N1*N2*N3)
            prob = np.exp(-df.flatten() / scale)
            prob = prob / prob.sum()
            indices_rnd = np.random.choice(indices, size=npoints, p=prob)
            idx1, idx2, idx3 = np.unravel_index(indices_rnd, [N1, N2, N3])
            idx1 = idx1 + np.random.rand(npoints)
            idx2 = idx2 + np.random.rand(npoints)
            idx3 = idx3 + np.random.rand(npoints)
            points = np.stack([idx1 / N1, idx2 / N2, idx3 / N3], axis=1)

        points -= 0.5

        return points


if __name__ == '__main__':
    app = Fusion()
    app.run()
