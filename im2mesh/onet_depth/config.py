import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.onet_depth import models, training, generation
from im2mesh.onet_depth.models import depth_predict_net
from im2mesh import data
from im2mesh import config

def get_depth_predictor(cfg):
    if 'depth_predictor' in cfg['model']:
        predictor = cfg['model']['depth_predictor']
    else:
        predictor = 'hourglass'

    if predictor == 'hourglass':
        dim = cfg['data']['dim']
        depth_predictor = depth_predict_net.DepthPredictNet(n_hourglass=1, img_dim=dim)
    elif predictor == 'uresnet' or predictor == 'uresnet18':
        pred_min_max = cfg['model']['pred_minmax']
        depth_predictor = depth_predict_net.UResnet_DepthPredict(pred_min_max=pred_min_max, num_layers=18)
    elif predictor == 'uresnet34':
        pred_min_max = cfg['model']['pred_minmax']
        depth_predictor = depth_predict_net.UResnet_DepthPredict(pred_min_max=pred_min_max, num_layers=34)
    elif predictor == 'uresnet50':
        pred_min_max = cfg['model']['pred_minmax']
        depth_predictor = depth_predict_net.UResnet_DepthPredict(pred_min_max=pred_min_max, num_layers=50)
    else:
        depth_predictor = None
        raise NotImplementedError

    return depth_predictor


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the OccupancyWithDepth Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    training_phase = cfg['training']['phase'] # 1 for depth prediction; 2 for reconstruction
    training_detach = cfg['training']['detach'] # detach or not
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim'] # should be 0
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']
    input_type = cfg['data']['input_type']

    p0_z = get_prior_z(cfg, device)
    if training_phase == 1:
        depth_predictor = get_depth_predictor(cfg)
        decoder = None
        encoder = None
        encoder_latent = None

        model = models.OccupancyWithDepthNetwork(
            depth_predictor, decoder, encoder, encoder_latent, p0_z=p0_z, device=device, 
        )
    else:
        if input_type == 'img_with_depth':
            depth_predictor = get_depth_predictor(cfg)
        elif input_type in ('depth_pred','depth_pointcloud','depth_pointcloud_completion'):
            depth_predictor = None
        else:
            raise NotImplementedError

        if 'use_local_feature' in cfg['model']:
            use_local_feature = cfg['model']['use_local_feature']
        else:
            use_local_feature = False
        
        if use_local_feature:
            decoder_local = cfg['model']['decoder_local']
            decoder_local_kwargs = cfg['model']['decoder_local_kwargs']
            decoder_local = models.decoder_local_dict[decoder_local](
                dim=dim, z_dim=z_dim, c_dim=cfg['model']['local_feature_dim'],
                **decoder_local_kwargs
            )
        else:
            decoder_local = None
        
        if 'logits1_ratio' in cfg['model']:
            local_logit_ratio = cfg['model']['logits1_ratio']
        else:
            local_logit_ratio = 1.
        
        decoder = models.decoder_dict[decoder](
            dim=dim, z_dim=z_dim, c_dim=c_dim,
            **decoder_kwargs
        )

        if z_dim != 0:
            encoder_latent = models.encoder_latent_dict[encoder_latent](
                dim=1, z_dim=z_dim, c_dim=c_dim,
                **encoder_latent_kwargs
            )
        else:
            encoder_latent = None

        if encoder == 'idx':
            encoder = nn.Embedding(len(dataset), c_dim)
        elif encoder is not None:
            if use_local_feature:
                encoder = encoder_dict[encoder](
                    c_dim=c_dim,
                    local=True,
                    local_feature_dim=cfg['model']['local_feature_dim'],
                    **encoder_kwargs
                )
            else:
                encoder = encoder_dict[encoder](
                    c_dim=c_dim,
                    **encoder_kwargs
                )
        else:
            encoder = None

        if 'space_carver_mode' in cfg['model']:
            space_carver_mode = cfg['model']['space_carver_mode']
        else:
            space_carver_mode = None

        if 'space_carver_eps' in cfg['model']:
            space_carver_eps = cfg['model']['space_carver_eps']
        else:
            space_carver_eps = None

        if 'space_carver_drop_p' in cfg['model']:
            space_carver_drop_p = cfg['model']['space_carver_drop_p']
        else:
            space_carver_drop_p = None

        
        model = models.OccupancyWithDepthNetwork(
            depth_predictor, decoder, encoder, encoder_latent, p0_z=p0_z, device=device, 
            decoder_local=decoder_local,
            local_logit_ratio=local_logit_ratio,
            space_carver_mode=space_carver_mode,
            space_carver_eps=space_carver_eps,
            space_carver_drop_p=space_carver_drop_p
        )

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
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    
    training_phase = cfg['training']['phase'] # 1 for depth prediction; 2 for reconstruction

    if training_phase == 1:
        if 'pred_minmax' in cfg['model']:
            pred_minmax = cfg['model']['pred_minmax']
        else:
            pred_minmax = False

        trainer = training.Phase1Trainer(
            model, optimizer,
            device=device, input_type=input_type,
            vis_dir=vis_dir,
            pred_minmax=pred_minmax
        )
    else:
        trainer_params = {
            'device':device, 'input_type':input_type,
            'vis_dir':vis_dir, 'threshold':threshold,
            'eval_sample':cfg['training']['eval_sample'],
        }

        if 'surface_loss_weight' in cfg['model']:
            trainer_params['surface_loss_weight'] = cfg['model']['surface_loss_weight']

        if ('loss_tolerance_episolon' in cfg['training']) and (0 in cfg['training']['loss_tolerance_episolon']):
            trainer_params['loss_tolerance_episolon'] = cfg['training']['loss_tolerance_episolon'][0]

        if ('sign_lambda' in cfg['training']) and (0 in cfg['training']['sign_lambda']):
            trainer_params['sign_lambda'] = cfg['training']['sign_lambda'][0]

        if 'loss_type' in cfg['training']:
            trainer_params['loss_type'] = cfg['training']['loss_type']

        if 'depth_map_mix' in cfg['training']:
            trainer_params['depth_map_mix'] = cfg['training']['depth_map_mix']

        if 'use_local_feature' in cfg['model']:
            trainer_params['local'] = cfg['model']['use_local_feature']

        if input_type == 'img_with_depth':
            trainer_params['training_detach'] = cfg['training']['detach'] # detach or not
            trainer = training.Phase2Trainer(
                model, optimizer,
                **trainer_params
            )
        elif input_type == 'depth_pred':
            if 'use_gt_depth' in cfg['training']:
                trainer_params['use_gt_depth_map'] = cfg['training']['use_gt_depth']

            if 'pred_with_img' in cfg['model']:
                trainer_params['with_img'] = cfg['model']['pred_with_img']
            
            trainer = training.Phase2HalfwayTrainer(
                model, optimizer,
                **trainer_params
            )
        elif input_type in ('depth_pointcloud', 'depth_pointcloud_completion'):
            if 'depth_pointcloud_transfer' in cfg['model']:
                trainer_params['depth_pointcloud_transfer'] = cfg['model']['depth_pointcloud_transfer']

            trainer = training.Phase2HalfwayTrainer(
                model, optimizer,
                **trainer_params
            )
        else:
            raise NotImplementedError('unsupported input_type for phase2,(only support img_with_depth & depth_pred)')
    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)
    input_type = cfg['data']['input_type']

    generator_params = {
        'device': device,
        'threshold': cfg['test']['threshold'],
        'resolution0': cfg['generation']['resolution_0'],
        'upsampling_steps': cfg['generation']['upsampling_steps'],
        'sample': cfg['generation']['use_sampling'],
        'refinement_step' :cfg['generation']['refinement_step'],
        'simplify_nfaces' :cfg['generation']['simplify_nfaces'],
        'preprocessor' :preprocessor,
        'input_type' :input_type,
    }

    if input_type == 'depth_pred':
        generator_params['use_gt_depth_map'] = cfg['training']['use_gt_depth']

    if 'pred_with_img' in cfg['model']:
        generator_params['with_img'] = cfg['model']['pred_with_img']

    if 'depth_pointcloud_transfer' in cfg['model']:
        generator_params['depth_pointcloud_transfer'] = cfg['model']['depth_pointcloud_transfer']

    if 'use_local_feature' in cfg['model']:
        generator_params['local'] = cfg['model']['use_local_feature']

    if 'use_occ_in_marching_cubes' in cfg['generation']:
        generator_params['use_occ_in_marching_cubes'] = cfg['generation']['use_occ_in_marching_cubes']

    if 'fixed_marching_cubes_threshold' in cfg['generation']:
        generator_params['fixed_marching_cubes_threshold'] = cfg['generation']['fixed_marching_cubes_threshold']

    generator = generation.Generator3D(
        model,
        **generator_params
    )
    return generator


from im2mesh.onet.config import get_prior_z 


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    training_phase = cfg['training']['phase'] # 1 for depth prediction; 2 for reconstruction

    fields = {}

    if training_phase == 1:
        pass
    else:
        from im2mesh.onet.config import get_data_fields as _get_data_fields
        return _get_data_fields(mode, cfg)

    return fields
