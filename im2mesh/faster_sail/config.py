from im2mesh import encoder
import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.faster_sail.models.sal import SALNetwork, decoder_dict as sal_decoder_dict, \
     encoder_latent_dict as sal_encoder_latent_dict
from im2mesh.faster_sail.models.sail_s3 import SAIL_S3Network, decoder_dict as sail_s3_decoder_dict
from im2mesh.faster_sail import training, generation
from im2mesh import data
from im2mesh.data.fields import PointsSALField
from im2mesh.data.transforms import SubsamplePointsSAL
from im2mesh import config


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    decoder = cfg['model']['decoder']
    encoder_latent = cfg['model']['encoder_latent']
    z_dim = cfg['model']['z_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']
    if method == 'SAL':
        #assert decoder == 'deepsdf'
        #assert encoder_latent == 'pointnet_vae'
        decoder = sal_decoder_dict[decoder](
            latent_size=z_dim, 
            **decoder_kwargs
        )

        if encoder_latent is not None:
            encoder_latent = sal_encoder_latent_dict[encoder_latent](
                c_dim=z_dim, dim=3,
                **encoder_latent_kwargs
            )

        model = SALNetwork(
            decoder, encoder_latent, device=device, z_dim=z_dim
        )        
    elif method == 'SAIL_S3':
        decoder = sail_s3_decoder_dict[decoder](
            latent_size=z_dim, 
            **decoder_kwargs
        )

        assert encoder_latent is None

        model = SAIL_S3Network(
            decoder, encoder_latent, device=device, z_dim=z_dim
        ) 
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
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')

    encoder_latent = cfg['model']['encoder_latent']

    trainer_params = {
        'device': device, 
        'vis_dir': vis_dir,
    }

    if method == 'SAL':
        with_encoder = (encoder_latent is not None)
        if with_encoder:
            optim_z_dim = 0
        else:
            optim_z_dim = cfg['model']['z_dim']

        trainer_params['with_encoder'] = with_encoder
        trainer_params['optim_z_dim'] = optim_z_dim
        if 'z_learning_rate' in cfg['training']:
            trainer_params['z_learning_rate'] = cfg['training']['z_learning_rate']
        else:
            trainer_params['z_learning_rate'] = 1e-4

        trainer = training.SALTrainer(
            model, optimizer, **trainer_params
        )
    elif method == 'SAIL_S3':
        if 'neighbors_K' in cfg['model']:
            trainer_params['K'] = cfg['model']['neighbors_K']

        if 'initial_length_alpha' in cfg['model']:
            trainer_params['initial_length_alpha'] = cfg['model']['initial_length_alpha']

        if 'initial_z_std' in cfg['model']:
            trainer_params['initial_z_std'] = cfg['model']['initial_z_std']

        trainer_params['optim_z_dim'] = cfg['model']['z_dim']
        if 'z_learning_rate' in cfg['training']:
            trainer_params['z_learning_rate'] = cfg['training']['z_learning_rate']
        else:
            trainer_params['z_learning_rate'] = 1e-3

        if 'refine_subfield' in cfg['model']:
            trainer_params['refine_subfield'] = cfg['model']['refine_subfield']

        if 'subfield_learning_rate' in cfg['training']:
            trainer_params['subfield_learning_rate'] = cfg['training']['subfield_learning_rate']

        if 'initial_center_func' in cfg['model']:
            trainer_params['initial_center_func'] = cfg['model']['initial_center_func']
            
        if 'use_subfield_weight' in cfg['training']:
            trainer_params['use_subfield_weight'] = cfg['training']['use_subfield_weight']

        if 'initial_r_t_func' in cfg['model']:
            trainer_params['initial_r_t_func'] = cfg['model']['initial_r_t_func']

        if 'initial_refine_subfield_centers' in cfg['model']:
            trainer_params['initial_refine_subfield_centers'] = cfg['model']['initial_refine_subfield_centers']

        trainer = training.SAIL_S3_Trainer(
            model, optimizer, **trainer_params
        )
    else:
        raise NotImplementedError

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)
    method = cfg['method']

    optim_z_dim = cfg['model']['z_dim']
    if method == 'SAL':
        generator_params = {
            'threshold': 0,
            'device': device,
            'resolution0': cfg['generation']['resolution_0'],
            'upsampling_steps': cfg['generation']['upsampling_steps'],
            'sample': cfg['generation']['use_sampling'],
            'refinement_step': cfg['generation']['refinement_step'],
            'simplify_nfaces': cfg['generation']['simplify_nfaces'],
            'preprocessor': preprocessor,
        }
        if 'furthur_refine' in cfg['generation']:
            furthur_refine = cfg['generation']['furthur_refine']
        else:
            furthur_refine = True
        generator = generation.SALGenerator(
            model,
            optim_z_dim=optim_z_dim,
            with_encoder=(cfg['model']['encoder_latent'] is not None),
            z_learning_rate=0.0001,
            z_refine_steps=20,
            furthur_refine=furthur_refine,
            **generator_params
        )
    elif method == 'SAIL_S3':
        generator_params = {
            'threshold': 0,
            'device': device,
            'resolution': cfg['generation']['resolution_0'],
            'simplify_nfaces': cfg['generation']['simplify_nfaces'],
            'preprocessor': preprocessor,
            'optim_z_dim': optim_z_dim
        }
        if 'interpolation_method' in cfg['generation']:
            generator_params['interpolation_method'] = cfg['generation']['interpolation_method']

        if 'interpolation_aggregate' in cfg['generation']:
            generator_params['interpolation_aggregate'] = cfg['generation']['interpolation_aggregate']

        if 'sign_decide_function' in cfg['generation']:
            generator_params['sign_decide_function'] = cfg['generation']['sign_decide_function']

        if 'voxelized_tolerance' in cfg['training']:
            generator_params['voxelized_tolerance'] = cfg['training']['voxelized_tolerance']

        trainer = get_trainer(model, None, cfg, device)

        generator = generation.SAIL_S3Generator(trainer, **generator_params)
    else:
        raise NotImplementedError

    return generator


def get_data_fields(mode, cfg):
    dataset_type = cfg['data']['dataset']
    # SAL dataset
    fields = {}
    N = cfg['data']['points_subsample']
    with_transforms = cfg['model']['use_camera']

    if 'points_subsample' in cfg['data'] and cfg['data']['points_subsample'] is not None: 
        points_transform = SubsamplePointsSAL(cfg['data']['points_subsample'])
    else:
        points_transform = None
    
    fields = {}
    points_file = cfg['data']['points_file']

    fields['points'] = PointsSALField(
        points_file, transform=points_transform,
        with_transforms=with_transforms,
    )

    return fields



