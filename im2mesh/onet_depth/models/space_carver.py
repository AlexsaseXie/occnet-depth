import torch
from torch import nn
from torch.nn import functional as F
from im2mesh.common import project_to_camera, transform_points
from torch.autograd import Function

class SpaceCarver(Function):
    '''
        Space Carver used in onet

    Args: carving mode : 'mask' or 'depth'
    Warning: shouldn't be used during backward 
    '''

    @staticmethod
    def forward(ctx, query_pts_img, reference, 
        world_mat=None, camera_mat=None, 
        mode='mask', eps=1e-3):
        '''
            returns the idx that is carved
        '''
        assert mode in ('mask', 'depth')

        if mode == 'mask':
            if world_mat is not None:
                query_pts_img = transform_points(query_pts_img, world_mat)

            if camera_mat is not None:
                query_pts_img = project_to_camera(query_pts_img, camera_mat)

            if query_pts_img.dim() == 3:
                query_pts_img = query_pts_img.unsqueeze(1) # B * 1 * n_pts * 2

            cor_mask = F.grid_sample(reference, query_pts_img, mode='nearest')
            cor_mask = cor_mask.reshape(cor_mask.size(0), cor_mask.size(3)) # B * n_pts
            remove_idx_bool = cor_mask < 1. - eps
        elif mode == 'depth':
            if world_mat is not None:
                query_pts_img = transform_points(query_pts_img, world_mat)

            assert camera_mat is not None
            # need z values
            z = query_pts_img[:,:,2] # B * n_pts

            query_pts_img = project_to_camera(query_pts_img, camera_mat)
            query_pts_img = query_pts_img.unsqueeze(1) # B * 1 * n_pts * 2

            cor_z = F.grid_sample(reference, query_pts_img, mode='nearest')
            cor_z = cor_z.reshape(cor_z.size(0), cor_z.size(3)) # B * n_pts
            remove_idx_bool = z < cor_z - eps
        else:
            raise NotImplementedError

        ctx.mark_non_differentiable(remove_idx_bool)
        return remove_idx_bool

    @staticmethod
    def backward(ctx, grad_idx_bool):
        return ()

space_carver = SpaceCarver.apply

class SpaceCarverModule(nn.Module):
    '''
        Space Carver Module
    Arg:
        mode
        eps

    Warning: shouldn't be used during backward
    '''
    def __init__(self, mode='mask', eps=3e-2):
        super(SpaceCarverModule, self).__init__()
        self.mode = mode
        self.eps = eps
        assert self.mode in ('mask', 'depth')

    def forward(self, query_pts, reference, cor_occ=None, world_mat=None, camera_mat=None):
        '''
            during training phase, this function will modify query_pts & cor_occ by inplace operations
        '''
        remove_idx_bool = space_carver(query_pts, reference, 
            world_mat=world_mat, camera_mat=camera_mat, 
            mode=self.mode, eps=self.eps
        )

        if self.training:
            # training phase behaviour
            assert cor_occ is not None
            # need to consider BN layers
            batch_size = query_pts.size(0)
            #TODO: remove this for
            for i in range(batch_size):
                # replace with negative points
                batch_remove_idx_bool = remove_idx_bool[i, :] # 1D: n_pts
                preserve_idx = torch.nonzero(batch_remove_idx_bool == 0).squeeze(1) # 1D: preserve_count
                remove_count = batch_remove_idx_bool.size(0) - preserve_idx.size(0)
                if remove_count == 0:
                    continue

                # random choose negative pts 
                exchange_idx = torch.randint(0, preserve_idx, (remove_count,)).long() # 1D: remove_count
                exchange_idx = torch.index_select(preserve_idx, 0, exchange_idx) # 1D: remove_count

                # replace ( inplace operation )
                query_pts[i, batch_remove_idx_bool, :] = torch.index_select(query_pts[i], 0, exchange_idx)
                cor_occ[i, batch_remove_idx_bool] = torch.index_select(cor_occ[i], 0, exchange_idx)
        else:
            # eval phase behaviour
            pass
            
        return remove_idx_bool