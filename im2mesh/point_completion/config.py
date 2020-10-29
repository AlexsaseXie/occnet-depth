import torch
from torchvision import transforms
from torch import nn
import os
from im2mesh import data
from im2mesh import config
from im2mesh.encoder import encoder_dict
from im2mesh.point_completion import model as pc_model
from im2mesh.point_completion import training
from im2mesh.encoder.world_mat_encoder import WorldMatEncoder


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the OccupancyWithDepth Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    encoder = cfg['model']['encoder']
    assert encoder.startswith('point')

    c_dim = cfg['model']['c_dim']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    if 'input_points_count' in cfg['model']:
        input_points_count = cfg['model']['input_points_count']
    else:
        input_points_count = 2048

    if 'output_points_count' in cfg['model']:
        output_points_count = cfg['model']['output_points_count']
    else:
        output_points_count = 2048

    if 'preserve_input' in cfg['model']:
        preserve_input = cfg['model']['preserve_input']
    else:
        preserve_input = False

    encoder = encoder_dict[encoder](
        c_dim=c_dim,
        **encoder_kwargs
    )

    if 'use_encoder_world_mat' in cfg['model']:
        use_encoder_world_mat = cfg['model']['use_encoder_world_mat']
    else:
        use_encoder_world_mat = False
    
    if use_encoder_world_mat:
        encoder_world_mat = WorldMatEncoder(c_dim=c_dim)
    else:
        encoder_world_mat = None

    model = pc_model.PointCompletionNetwork(encoder, device=device, c_dim=c_dim,
        input_points_count=input_points_count, 
        output_points_count=output_points_count, 
        preserve_input=preserve_input,
        encoder_world_mat=encoder_world_mat
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    
    trainer_params = {}

    if 'depth_pointcloud_transfer' in cfg['model']:
        trainer_params['depth_pointcloud_transfer'] = cfg['model']['depth_pointcloud_transfer']

    if 'gt_pointcloud_transfer' in cfg['model']:
        trainer_params['gt_pointcloud_transfer'] = cfg['model']['gt_pointcloud_transfer']

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
    return trainer


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
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
