import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
from im2mesh.utils.lib_pointcloud_voxel import grid_points_query_range
import time
from torch.nn import functional as F
from tqdm import tqdm


class SALGenerator(object):
    '''  Generator class for SAL Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 simplify_nfaces=None,
                 preprocessor=None, with_encoder=True, optim_z_dim=128,
                 furthur_refine=True,
                 z_learning_rate=1e-4, z_refine_steps=20
                 ):
        self.model = model.to(device)
        if getattr(self.model, 'module', False):
            # force to use single gpu forward
            self.model = self.model.module
        self.with_encoder = with_encoder
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor

        self.optim_z_dim = optim_z_dim
        self.z_learning_rate = z_learning_rate
        self.z_refine_steps = z_refine_steps
        self.furthur_refine = furthur_refine

    def generate_mesh(self, data, return_stats=True, z_prior=None):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        tmp_stats_dict = {}

        if self.with_encoder:
            inputs = data.get('inputs', torch.empty(1, 0)).to(device)
            with torch.no_grad():
                q_z, z_reg = self.model.infer_z(inputs)
                z = q_z.rsample()
        else:
            with torch.no_grad():
                if z_prior is not None:
                    z = z_prior.to(device).detach()
                else:
                    # batch size == 1
                    z = torch.randn(1, self.optim_z_dim).to(device).detach()

        kwargs = {}

        mesh = self.generate_from_latent(z, stats_dict=tmp_stats_dict, **kwargs)
        for key in tmp_stats_dict:
            stats_dict['(before z refine) %s' % key] = tmp_stats_dict[key]

        # furthur refine 
        if self.furthur_refine:
            z = z.requires_grad_()
            z_optimizer = optim.SGD([z], lr=self.z_learning_rate)

            scheduler = optim.lr_scheduler.StepLR(z_optimizer, step_size=self.z_refine_steps//2.5, gamma=0.1)
            p = data.get('points').to(device)
            gt_sal_val = data.get('points.sal').to(device)

            t0 = time.time()
            for i in range(self.z_refine_steps):
                z_optimizer.zero_grad()
                loss = self.model(p, func='z_loss',
                    gt_sal=gt_sal_val, z_loss_ratio=1.0e-3, z=z)

                loss.backward()
                z_optimizer.step()
                scheduler.step()

            z = z.clone().detach()
            stats_dict['time (refine z)'] = time.time() - t0

            refined_mesh = self.generate_from_latent(z, stats_dict=stats_dict, **kwargs)

        if return_stats:
            if self.furthur_refine:
                return mesh, refined_mesh, stats_dict
            else:
                return mesh, stats_dict
        else:
            if self.furthur_refine:
                return mesh, refined_mesh
            else:
                return mesh

    def generate_from_latent(self, z, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = self.threshold

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, z, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, z, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, z, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, z, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        sdf_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                sdf_hat = self.model.decode(pi, z, **kwargs)

            sdf_hats.append(sdf_hat.squeeze(0).detach().cpu())

        sdf_hat = torch.cat(sdf_hats, dim=0)

        return sdf_hat

    def extract_mesh(self, sdf_hat, z, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            sdf_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = sdf_hat.shape
        box_size = 1 + self.padding
        threshold = self.threshold
        # Make sure that mesh is watertight
        t0 = time.time()
        #sdf_hat_padded = sdf_hat
        sdf_hat_padded = np.pad(
           sdf_hat, 1, 'constant', constant_values=1e6)
        vertices, triangles = libmcubes.marching_cubes(
            sdf_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, z)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, sdf_hat, z)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, z):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        z, c = z.unsqueeze(0), c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            sdf_hat = self.model.decode(vi, z)
            out = sdf_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, sdf_hat, z, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            sdf_hat (tensor): predicted occupancy grid
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = sdf_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), z, c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh


def get_grid_idx_float(x, N, low=-0.5, high=0.5):
    grid_len = (high - low) / N
    idx = (x - low) / grid_len
    return idx

def x_intersect(x_low1, x_high1, x_low2, x_high2):
    data_list = [(x_low1, 0), (x_high1, 0), (x_low2, 1), (x_high2, 1)]
    data_list.sort(key=lambda x: x[0])
    if data_list[0][1] != data_list[1][1]:
        return [data_list[1][0], data_list[2][0]]
    else:
        return False

def xyz_intersect(xyz_low1, xyz_high1, xyz_low2, xyz_high2):
    x_range = x_intersect(xyz_low1[0], xyz_high1[0], xyz_low2[0], xyz_high2[0])
    y_range = x_intersect(xyz_low1[1], xyz_high1[1], xyz_low2[1], xyz_high2[1])
    z_range = x_intersect(xyz_low1[2], xyz_high1[2], xyz_low2[2], xyz_high2[2])

    if x_range != False and y_range != False and z_range != False:
        return [x_range, y_range, z_range]
    else:
        return False


N_SAMPLE = 2000

class SAIL_S3Generator(object):
    def __init__(self, trainer, points_batch_size=100000,
                 threshold=0,  device=None,
                 resolution=256, padding=0.1, 
                 simplify_nfaces=None,
                 preprocessor=None, optim_z_dim=128,
                 interpolation_method='sail_s3_paper',
                 interpolation_aggregate='mean',
                 sign_decide_function='prim'
                 ):
        self.trainer = trainer
        
        self.points_batch_size = points_batch_size
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.padding = padding
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor

        self.optim_z_dim = optim_z_dim
        self.furthur_refine = False

        self.K = self.trainer.K # K subfields
        self.hj = None # hj function 
        self.interpolation_method = interpolation_method
        self.interpolation_aggregate = interpolation_aggregate
        self.sign_decide_function = sign_decide_function

        print('Interpolation method:', self.interpolation_method, ',aggregate:', self.interpolation_aggregate)
        print('Sign decide function:', self.sign_decide_function)

    def generate_mesh(self, data, out_dict, return_stats=True, separate=False):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.trainer.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}

        for key in out_dict:
            data[key] = out_dict[key]
        self.trainer.init_z(data, initialize_optimizer=False)
        self.out_dict = out_dict

        pc_numpy = data.get('inputs.pointcloud').squeeze(0).numpy()

        self.center = self.out_dict['center'].squeeze(0).numpy() # K * 3
        self.length = self.out_dict['length'].squeeze(0).numpy() # K

        print('Mean length:', self.length.mean())

        high = (1 + self.padding) / 2.0
        low = -high

        #tolerance_range = min((1 / self.resolution) * 3, 0.008)
        tolerance_range = 0.005
        self.voxel, self.calc_index, self.inside_index, self.outside_index = grid_points_query_range(
            pc_numpy, 
            self.resolution, tolerance_range, 
            low, high
        )

        print('Grid points query results. voxel:', self.voxel.shape, 'calc_index:', self.calc_index.shape)
        print('outside_index:', self.outside_index.shape, 'inside_index:', self.inside_index.shape)

        if self.sign_decide_function == 'prim':
            self.prim_decide_subfield_sign()
        elif self.sign_decide_function == 'simple':
            self.simple_decide_subfield_sign()
        else:
            raise NotImplementedError

        if not separate:
            mesh = self.generate_from_latent(stats_dict=stats_dict, **kwargs)

            if return_stats:
                return mesh, stats_dict
            else:
                return mesh
        else:
            return self.generate_from_latent_separate()         

    def _decide_sign_for_id(self, initial_id):
        center = self.center
        length = self.length
        initial_center = center[initial_id, :] # (3,)
        initial_length = length[initial_id]

        xyz_low = initial_center - initial_length / 2.0
        xyz_high = initial_center + initial_length / 2.0

        xyz_low = np.round(get_grid_idx_float(xyz_low, self.resolution))
        xyz_low[xyz_low < 0] = 0
        xyz_low[xyz_low >= self.resolution - 1] = self.resolution - 1
        xyz_high = np.round(get_grid_idx_float(xyz_high, self.resolution))
        xyz_high[xyz_high < 0] = 0
        xyz_high[xyz_high >= self.resolution - 1] = self.resolution - 1

        #print('Initial xyz:', xyz_low, xyz_high)
        need_calc_voxels = self.voxel[int(xyz_low[0]):int(xyz_high[0])+1, 
            int(xyz_low[1]):int(xyz_high[1])+1, 
            int(xyz_low[2]):int(xyz_high[2])+1, :]
        outside_points = need_calc_voxels[need_calc_voxels[:,:,:,3] == 0][:,:3] # a1 * 3
        inside_points = need_calc_voxels[need_calc_voxels[:,:,:,3] == 2][:,:3] # a2 * 3
        
        if outside_points.shape[0] > N_SAMPLE:
            rand_idx = np.random.choice(outside_points.shape[0], size=N_SAMPLE, replace=False)
            outside_points = outside_points[rand_idx,:]
        if inside_points.shape[0] > N_SAMPLE:
            rand_idx = np.random.choice(inside_points.shape[0], size=N_SAMPLE, replace=False)
            inside_points = inside_points[rand_idx,:]

        outside_points_torch = torch.from_numpy(outside_points).unsqueeze(0) # 1 * N_SAMPLE * 3
        inside_points_torch = torch.from_numpy(inside_points).unsqueeze(0) # 1 * N_SAMPLE * 3

        if outside_points_torch.shape[1] != 0:
            w_outside = initial_length - (outside_points_torch - initial_center).abs().max(axis=2)[0] # 1 * N_SAMPLE
        if inside_points_torch.shape[1] != 0:
            w_inside  = initial_length - (inside_points_torch - initial_center).abs().max(axis=2)[0] # 1 * N_SAMPLE

        # print('_decide sign outside points.shape:', outside_points_torch.shape)
        # print('_decide sign inside points.shape:', inside_points_torch.shape)
        with torch.no_grad():
            outside_p_r = self.trainer.predict_for_points_with_specific_subfield(outside_points_torch, initial_id)
            inside_p_r = self.trainer.predict_for_points_with_specific_subfield(inside_points_torch, initial_id)

            if outside_points_torch.shape[1] != 0:
                outside_p_r = outside_p_r * w_outside.to(self.device)
            if inside_points_torch.shape[1] != 0:
                inside_p_r = inside_p_r * w_inside.to(self.device)
            loss_positive = F.relu(-outside_p_r).sum() + F.relu(inside_p_r).sum()
            loss_negative = F.relu(outside_p_r).sum() + F.relu(-inside_p_r).sum()

        if loss_positive < loss_negative:
            return True
        else:
            print('---Loss +:', loss_positive, '-:', loss_negative)
            return False

    def simple_decide_subfield_sign(self):
        hj = np.ones((self.K), dtype=np.bool)

        print('Simple decide subfields sign')
        for i in range(self.K):
            hj[i] = self._decide_sign_for_id(i)

        self.hj = hj.astype(np.float)
        self.hj[self.hj==0] = -1
        #print('Hj:', self.hj)

    def prim_decide_subfield_sign(self):
        initial_id = np.random.randint(self.K)

        print("Deciding the hj sign...")

        hj = np.ones((self.K), dtype=np.bool)
        # decide the first sign
        center = self.center
        length = self.length
        hj[initial_id] = self._decide_sign_for_id(initial_id)

        print('Finish deciding the initial sign...')

        # prim algorithm
        visited = np.zeros((self.K), dtype=np.bool)
        dis = np.empty((self.K), dtype=np.float32)
        dis[:] = 1e6
        potential_hj = np.zeros((self.K), dtype=np.bool)
        current_subset_count = 1
        current_id = initial_id
        visited[initial_id] = True
        
        def update_and_choose(cur_id):
            cur_xyz_low = center[cur_id] - length[cur_id]
            cur_xyz_high = center[cur_id] + length[cur_id]

            nearest_id = -1
            nearest_dis = 1e6

            for i in range(self.K):
                if visited[i] or i == cur_id:
                    continue
                
                tar_xyz_low = center[i] - length[i]
                tar_xyz_high = center[i] + length[i]

                intersect_range = xyz_intersect(cur_xyz_low, cur_xyz_high, tar_xyz_low, tar_xyz_high)
                if intersect_range != False:
                    sample_points = np.random.rand(1, N_SAMPLE, 3).astype(np.float32)
                    for j in range(3):
                        sample_points[:,:,j] *= (intersect_range[j][1] - intersect_range[j][0])
                        sample_points[:,:,j] += intersect_range[j][0]
                    
                    sample_points_torch = torch.from_numpy(sample_points)
                    cur_p_r = self.trainer.predict_for_points_with_specific_subfield(sample_points_torch, cur_id)
                    tar_p_r = self.trainer.predict_for_points_with_specific_subfield(sample_points_torch, i)

                    loss_same = (cur_p_r - tar_p_r).abs().sum()
                    loss_oppo = (cur_p_r + tar_p_r).abs().sum()

                    if loss_same < loss_oppo and loss_same < dis[i]:
                        dis[i] = loss_same
                        potential_hj[i] = hj[cur_id]
                    elif loss_oppo <= loss_same and loss_oppo < dis[i]:
                        dis[i] = loss_oppo
                        potential_hj[i] = ~hj[cur_id]
                    
                if dis[i] < nearest_dis:
                    nearest_dis = dis[i]
                    nearest_id = i

            return nearest_id

        print('Prim algorithm...')
        while current_subset_count < self.K:
            current_id = update_and_choose(current_id)
            visited[current_id] = True
            hj[current_id] = potential_hj[current_id]
            current_subset_count += 1

        self.hj = hj.astype(np.float)
        self.hj[self.hj==0] = -1

        #print('Hj:', self.hj)

    def _genarate_mesh_by_id(self, initial_id):
        center = self.center
        length = self.length
        initial_center = center[initial_id, :] # (3,)
        initial_length = length[initial_id]

        xyz_low = initial_center - initial_length
        xyz_high = initial_center + initial_length

        xyz_low = np.round(get_grid_idx_float(xyz_low, self.resolution))
        xyz_low[xyz_low < 0] = 0
        xyz_low[xyz_low >= self.resolution - 1] = self.resolution - 1
        xyz_high = np.round(get_grid_idx_float(xyz_high, self.resolution))
        xyz_high[xyz_high < 0] = 0
        xyz_high[xyz_high >= self.resolution - 1] = self.resolution - 1

        #print('Initial xyz:', xyz_low, xyz_high)
        need_calc_voxels = self.voxel[int(xyz_low[0]):int(xyz_high[0])+1, 
            int(xyz_low[1]):int(xyz_high[1])+1, 
            int(xyz_low[2]):int(xyz_high[2])+1, :]

        N_X = need_calc_voxels.shape[0]
        N_Y = need_calc_voxels.shape[1]
        N_Z = need_calc_voxels.shape[2]

        box_low = need_calc_voxels[0,0,0,:3]
        box_high = need_calc_voxels[-1,-1,-1,:3]

        need_calc_voxels = need_calc_voxels.reshape(-1, 4)
        points_torch = torch.from_numpy(need_calc_voxels[:,:3]).unsqueeze(0)

        # print('N_X, N_Y, N_Z:', N_X, N_Y, N_Z)
        # print('point torch.shape:', points_torch.shape)
        
        p_r = self.trainer.predict_for_points_with_specific_subfield(points_torch, initial_id)

        sdf_hat = p_r.cpu().numpy().reshape(N_X, N_Y, N_Z)

        vertices, triangles = libmcubes.marching_cubes(
            sdf_hat, 0)
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Normalize to bounding box
        vertices /= np.array([N_X-1, N_Y-1, N_Z-1])
        vertices = vertices * (box_high - box_low) + box_low

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        return mesh

    def generate_from_latent_separate(self):
        meshes = []
        for i in tqdm(range(self.K)):
            mesh = self._genarate_mesh_by_id(i)

            meshes.append(mesh)

        return meshes

    def generate_from_latent(self, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = self.threshold

        print('Hj:', self.hj)

        t0 = time.time()
        # Compute bounding box size
        voxel_sdf_hat = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)
        voxel_sdf_hat[self.outside_index[:,0], self.outside_index[:,1], self.outside_index[:,2]] = 10
        voxel_sdf_hat[self.inside_index[:,0], self.inside_index[:,1], self.inside_index[:,2]] = -10

        ps = self.voxel[self.calc_index[:,0], self.calc_index[:,1], self.calc_index[:,2]][:,:3]
        ps = torch.from_numpy(ps).unsqueeze(0) # 1 * N * 3
        print('Need calc points:', ps.shape)
        #p_r = self.trainer.predict_for_points(ps, hj=self.hj)
        p_r = self.trainer.predict_for_points_fast(ps, hj=self.hj, 
            weight_func=self.interpolation_method, aggregate=self.interpolation_aggregate
        )
        
        print('Finish predict')
        voxel_sdf_hat[self.calc_index[:,0], self.calc_index[:,1], self.calc_index[:,2]] = p_r.numpy()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(voxel_sdf_hat, stats_dict=stats_dict)
        return mesh

    def extract_mesh(self, sdf_hat, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            sdf_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = sdf_hat.shape
        box_size = 1 + self.padding
        threshold = self.threshold
        # Make sure that mesh is watertight
        t0 = time.time()
        sdf_hat_padded = np.pad(
            sdf_hat, 1, 'constant', constant_values=1e6)
        vertices, triangles = libmcubes.marching_cubes(
            sdf_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        return mesh
