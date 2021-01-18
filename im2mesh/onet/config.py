import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.onet import models, training, generation
from im2mesh import data
from im2mesh import config


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']

    decoder = models.decoder_dict[decoder](
        dim=dim, z_dim=z_dim, c_dim=c_dim,
        **decoder_kwargs
    )

    if z_dim != 0:
        encoder_latent = models.encoder_latent_dict[encoder_latent](
            dim=dim, z_dim=z_dim, c_dim=c_dim,
            **encoder_latent_kwargs
        )
    else:
        encoder_latent = None

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim,
            **encoder_kwargs
        )
    else:
        encoder = None

    p0_z = get_prior_z(cfg, device)
    model = models.OccupancyNetwork(
        decoder, encoder, encoder_latent, p0_z, device=device
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
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    if 'surface_loss_weight' in cfg['model']:
        surface_loss_weight = cfg['model']['surface_loss_weight']
    else:
        surface_loss_weight = 1.

    if ('loss_tolerance_episolon' in cfg['training']) and (0 in cfg['training']['loss_tolerance_episolon']):
        loss_tolerance_episolon = cfg['training']['loss_tolerance_episolon'][0]
    else:
        loss_tolerance_episolon = 0.

    if ('sign_lambda' in cfg['training']) and (0 in cfg['training']['sign_lambda']):
        sign_lambda = cfg['training']['sign_lambda'][0]
    else:
        sign_lambda = 0.

    trainer_params = {
        'device': device, 'input_type': input_type,
        'vis_dir': vis_dir, 'threshold': threshold,
        'eval_sample': cfg['training']['eval_sample'],
        'surface_loss_weight': surface_loss_weight,
        'loss_tolerance_episolon': loss_tolerance_episolon,
        'sign_lambda': sign_lambda
    }

    if 'use_sdf' in cfg and cfg['use_sdf']:
        trainer_constructor = training.SDFTrainer
        if 'sdf_ratio' in cfg['model']:
            trainer_params['sdf_ratio'] = cfg['model']['sdf_ratio']
    else:
        trainer_constructor = training.Trainer

    trainer = trainer_constructor(
        model, optimizer,
        **trainer_params
    )

    if 'loss_type' in cfg['training']:
        trainer.loss_type = cfg['training']['loss_type']
        print('loss type:', trainer.loss_type)

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_sdf_data_fields(mode, cfg):
    # currently only support h5
    N = cfg['data']['points_subsample']
    with_transforms = cfg['model']['use_camera']
    if mode == 'train':
        if 'input_range' in cfg['data']:
            input_range = cfg['data']['input_range']
            print('Input range:', input_range)
        else:
            input_range = None
    else:
        if 'test_range' in cfg['data']:
            input_range = cfg['data']['test_range']
            print('Test range:', input_range)
        else:
            input_range = None

    fields = {}
    points_file = cfg['data']['points_file']
    fields['points'] = data.SdfH5Field(
        points_file, subsample_n=N,
        with_transforms=with_transforms,
        input_range=input_range
    )

    if mode in ('val', 'test'):
        if 'val_subsample' in cfg['test']:
            val_subsample = cfg['test']['val_subsample']
        else:
            val_subsample = None
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.SdfH5Field(
                points_iou_file, subsample_n=val_subsample, 
                with_transforms=with_transforms,
                input_range=input_range
            )

        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields


def get_occ_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    N = cfg['data']['points_subsample']
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']

    if mode == 'train':
        if 'input_range' in cfg['data']:
            input_range = cfg['data']['input_range']
            print('Input range:', input_range)
        else:
            input_range = None
    else:
        if 'test_range' in cfg['data']:
            input_range = cfg['data']['test_range']
            print('Test range:', input_range)
        else:
            input_range = None

    fields = {}
    points_file = cfg['data']['points_file']
    if points_file.endswith('.npz'):
        fields['points'] = data.PointsField(
            cfg['data']['points_file'], points_transform,
            with_transforms=with_transforms,
            unpackbits=cfg['data']['points_unpackbits'],
            input_range=input_range
        )
    elif points_file.endswith('.h5'):
        fields['points'] = data.PointsH5Field(
            cfg['data']['points_file'], subsample_n=N,
            with_transforms=with_transforms,
            input_range=input_range
        )
    else:
        raise NotImplementedError

    if mode in ('val', 'test'):
        if 'val_subsample' in cfg['test']:
            val_subsample = cfg['test']['val_subsample']
            val_subsample_transform = data.SubsamplePoints(val_subsample)
        else:
            val_subsample = None
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            if points_iou_file.endswith('.npz'):
                fields['points_iou'] = data.PointsField(
                    points_iou_file, val_subsample_transform,
                    with_transforms=with_transforms,
                    unpackbits=cfg['data']['points_unpackbits'],
                    input_range=input_range
                )
            elif points_iou_file.endswith('.h5'):
                fields['points_iou'] = data.PointsH5Field(
                    points_iou_file, subsample_n=val_subsample,
                    with_transforms=with_transforms,
                    input_range=input_range
                )
            else:
                raise NotImplementedError

        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields


def get_pix3d_data_fields(mode, cfg):
    fields = {}
    # TODO: get points.loc points.scale
    return fields


def get_data_fields(mode, cfg):
    dataset_type = cfg['data']['dataset']
    if dataset_type == 'pix3d':
        return get_pix3d_data_fields(mode, cfg)

    assert dataset_type == 'Shapes3D'

    if 'use_sdf' in cfg:
        use_sdf = cfg['use_sdf']
    else:
        use_sdf = False

    if use_sdf:
        return get_sdf_data_fields(mode, cfg)
    else:
        return get_occ_data_fields(mode, cfg)
