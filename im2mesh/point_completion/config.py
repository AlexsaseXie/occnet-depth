import torch
from torchvision import transforms
from torch import nn
import os
from im2mesh import data
from im2mesh import config
from im2mesh.encoder import encoder_dict
from im2mesh.point_completion.models import FCAE_model, MSN_model
from im2mesh.point_completion import training
from im2mesh.encoder.world_mat_encoder import WorldMatEncoder


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Point Completion Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    if method == 'point_completion':
        method = 'FCAE'

    # encoder
    encoder = cfg['model']['encoder']
    if method == 'MSN':
        # assert
        encoder = 'msn_pointnet'

    # c_dim
    c_dim = cfg['model']['c_dim']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    encoder = encoder_dict[encoder](
        c_dim=c_dim,
        **encoder_kwargs
    )

    if 'input_points_count' in cfg['model']:
        input_points_count = cfg['model']['input_points_count']
    else:
        input_points_count = 2048

    if 'output_points_count' in cfg['model']:
        output_points_count = cfg['model']['output_points_count']
    else:
        output_points_count = 2048

    if method == 'FCAE':
        if 'preserve_input' in cfg['model']:
            preserve_input = cfg['model']['preserve_input']
        else:
            preserve_input = False

        if 'use_encoder_world_mat' in cfg['model']:
            use_encoder_world_mat = cfg['model']['use_encoder_world_mat']
        else:
            use_encoder_world_mat = False
        
        if use_encoder_world_mat:
            encoder_world_mat = WorldMatEncoder(c_dim=c_dim)
        else:
            encoder_world_mat = None

        model = FCAE_model.PointCompletionNetwork(encoder, device=device, c_dim=c_dim,
            input_points_count=input_points_count, 
            output_points_count=output_points_count, 
            preserve_input=preserve_input,
            encoder_world_mat=encoder_world_mat
        )
    elif method == 'MSN':
        if 'n_primitives' in cfg['model']:
            n_primitives = cfg['model']['n_primitives']
        else:
            n_primitives = 16

        if 'space_carver_mode' in cfg['model']:
            space_carver_mode = cfg['model']['space_carver_mode']
        else:
            space_carver_mode = None

        if 'space_carver_eps' in cfg['model']:
            space_carver_eps = cfg['model']['space_carver_eps']
        else:
            space_carver_eps = None

        model = MSN_model.MSN(encoder, num_points = output_points_count, n_primitives = n_primitives,
                space_carver_mode=space_carver_mode, space_carver_eps=space_carver_eps)
    else:
        raise NotImplementedError

    if 'data_parallel' in cfg:
        data_parallel = cfg['data_parallel']
    else:
        data_parallel = None
    assert data_parallel in (None, 'DP', 'DDP')

    # using multiple gpu
    if data_parallel == 'DP':
        model = torch.nn.DataParallel(model)
    elif data_parallel == 'DDP':
        #TODO: construct DDP module
        raise NotImplementedError

    model = model.to(device)
    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    method = cfg['method']
    if method == 'point_completion':
        method = 'FCAE'

    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    
    trainer_params = {}

    if 'depth_pointcloud_transfer' in cfg['model']:
        trainer_params['depth_pointcloud_transfer'] = cfg['model']['depth_pointcloud_transfer']

    if 'gt_pointcloud_transfer' in cfg['model']:
        trainer_params['gt_pointcloud_transfer'] = cfg['model']['gt_pointcloud_transfer']

    if method == 'FCAE':
        if 'view_penalty' in cfg['training']:
            trainer_params['view_penalty'] = cfg['training']['view_penalty']
            
            if cfg['training']['view_penalty']:
                if 'loss_mask_flow_ratio' in cfg['training']:
                    trainer_params['loss_mask_flow_ratio'] = cfg['training']['loss_mask_flow_ratio']
                if 'mask_flow_eps' in cfg['training']:
                    trainer_params['mask_flow_eps'] = cfg['training']['mask_flow_eps']         
                if 'loss_depth_test_ratio' in cfg['training']:
                    trainer_params['loss_depth_test_ratio'] = cfg['training']['loss_depth_test_ratio']
                if 'depth_test_eps' in cfg['training']:
                    trainer_params['depth_test_eps'] = cfg['training']['depth_test_eps']

        if 'loss_type' in cfg['training']:
            assert cfg['training']['loss_type'] in ('cd', 'emd')
            trainer_params['loss_type'] = cfg['training']['loss_type']

        trainer = training.PointCompletionTrainer(model, optimizer,
            device=device, input_type=input_type,
            vis_dir=vis_dir, **trainer_params
        )
    elif method == 'MSN':
        trainer = training.MSNTrainer(model, optimizer,
            device=device, input_type=input_type,
            vis_dir=vis_dir, **trainer_params
        )
    else:
        raise NotImplementedError

    return trainer


def get_pix3d_data_fields(mode, cfg):
    fields = {}
    #TODO: add pix3d data field load
    return fields


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    dataset_type = cfg['data']['dataset']
    if dataset_type == 'pix3d':
        return get_pix3d_data_fields(mode, cfg)

    fields = {}
    if 'output_points_count' in cfg['model']:
        output_points_count = cfg['model']['output_points_count']
    else:
        output_points_count = 2048

    transform = transforms.Compose([
        data.SubsamplePointcloud(output_points_count)
    ])

    fields['pointcloud'] = data.PointCloudField(
        cfg['data']['pointcloud_file'], transform,
        with_transforms=True
    )
    
    return fields
