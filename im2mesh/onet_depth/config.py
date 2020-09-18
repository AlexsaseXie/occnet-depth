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
    elif predictor == 'uresnet':
        pred_min_max = cfg['model']['pred_minmax']
        depth_predictor = depth_predict_net.UResnet_DepthPredict(pred_min_max=pred_min_max)
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

    
    if training_phase == 1:
        depth_predictor = get_depth_predictor(cfg)
        decoder = None
        encoder = None
        encoder_latent = None
    else:
        if input_type == 'img_with_depth':
            depth_predictor = get_depth_predictor(cfg)
        elif input_type == 'depth_pred':
            depth_predictor = None
        else:
            raise NotImplementedError
        
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
            encoder = encoder_dict[encoder](
                c_dim=c_dim,
                **encoder_kwargs
            )
        else:
            encoder = None

    p0_z = get_prior_z(cfg, device)
    model = models.OccupancyWithDepthNetwork(
        depth_predictor, decoder, encoder, encoder_latent, p0_z=p0_z, device=device
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

        if 'loss_type' in cfg['training']:
            loss_type = cfg['training']['loss_type']
        else:
            loss_type = 'cross_entropy'

        if 'depth_map_mix' in cfg['training']:
            depth_map_mix = cfg['training']['depth_map_mix']
        else:
            depth_map_mix = False

        if input_type == 'img_with_depth':
            training_detach = cfg['training']['detach'] # detach or not
            trainer = training.Phase2Trainer(
                model, optimizer,
                device=device, input_type=input_type,
                vis_dir=vis_dir, threshold=threshold,
                eval_sample=cfg['training']['eval_sample'],
                loss_type=loss_type,
                surface_loss_weight=surface_loss_weight,
                loss_tolerance_episolon=loss_tolerance_episolon,
                sign_lambda=sign_lambda,
                training_detach=training_detach,
                depth_map_mix=depth_map_mix
            )
        elif input_type == 'depth_pred' or input_type == 'depth_pointcloud':
            training_use_gt_depth = cfg['training']['use_gt_depth']
            trainer = training.Phase2HalfwayTrainer(
                model, optimizer,
                device=device, input_type=input_type,
                vis_dir=vis_dir, threshold=threshold,
                eval_sample=cfg['training']['eval_sample'],
                loss_type=loss_type,
                surface_loss_weight=surface_loss_weight,
                loss_tolerance_episolon=loss_tolerance_episolon,
                sign_lambda=sign_lambda,
                use_gt_depth_map=training_use_gt_depth,
                depth_map_mix=depth_map_mix
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
    if input_type == 'depth_pred':
        training_use_gt_depth = cfg['training']['use_gt_depth']
    else:
        training_use_gt_depth = False

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
        input_type=input_type,
        use_gt_depth=training_use_gt_depth
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


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    N = cfg['data']['points_subsample']
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']
    training_phase = cfg['training']['phase'] # 1 for depth prediction; 2 for reconstruction

    fields = {}

    if training_phase == 1:
        pass
    else:
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

        points_file = cfg['data']['points_file']
        if points_file.endswith('.npz'):
            fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
                input_range=input_range,
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
            points_iou_file = cfg['data']['points_iou_file']
            voxels_file = cfg['data']['voxels_file']
            if points_iou_file is not None:
                if points_iou_file.endswith('.npz'):
                    fields['points_iou'] = data.PointsField(
                        points_iou_file,
                        with_transforms=with_transforms,
                        unpackbits=cfg['data']['points_unpackbits'],
                        input_range=input_range
                    )
                elif points_iou_file.endswith('.h5'):
                    fields['points_iou'] = data.PointsH5Field(
                        points_iou_file, 
                        with_transforms=with_transforms,
                        input_range=input_range
                    )
                else:
                    raise NotImplementedError

            if voxels_file is not None:
                fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
