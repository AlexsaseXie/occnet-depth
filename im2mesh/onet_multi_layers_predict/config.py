import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.onet_multi_layers_predict.models import feature_extractor
from im2mesh.onet_multi_layers_predict import models, training, generation, generation_local
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

    # model_type
    use_local_feature = cfg['model']['use_local_feature']
    if use_local_feature:
        decoder_local = cfg['model']['decoder_local']
        decoder_local_kwargs = cfg['model']['decoder_local_kwargs']

    decoder3 = models.decoder_dict[decoder](
        dim=dim, z_dim=z_dim, c_dim=c_dim,
        **decoder_kwargs
    )

    if use_local_feature:
        decoder2 = models.decoder_local_dict[decoder_local](
            dim=dim, z_dim=z_dim, c_dim=cfg['model']['local_feature_dim'],
            **decoder_local_kwargs
        )
        decoder1 = models.decoder_local_dict[decoder_local](
            dim=dim, z_dim=z_dim, c_dim=cfg['model']['local_feature_dim'],
            **decoder_local_kwargs
        )
    else:
        decoder2 = models.decoder_dict[decoder](
            dim=dim, z_dim=z_dim, c_dim=c_dim,
            **decoder_kwargs
        )
        decoder1 = models.decoder_dict[decoder](
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

    if encoder is not None:
        if use_local_feature:
            print('Using encoder:', encoder)
            if encoder == 'local_1':
                encoder = feature_extractor.Resnet18_Local_1(
                    c_dim=c_dim, feature_map_dim=cfg['model']['local_feature_dim'],
                    **encoder_kwargs
                )
            else:
                encoder = feature_extractor.Resnet18_Local(
                    c_dim=c_dim, feature_map_dim=cfg['model']['local_feature_dim'],
                    **encoder_kwargs
                )
        else:
            # encoder == 'full'
            encoder = feature_extractor.Resnet18_Full(
                c_dim=c_dim,
                **encoder_kwargs
            )
    else:
        encoder = None

    p0_z = get_prior_z(cfg, device)
    model = models.OccupancyNetwork(
        dataset, decoder1, decoder2, decoder3, encoder, encoder_latent, p0_z, device=device, use_local_feature=use_local_feature,
        logits2_ratio=cfg['model']['logits2_ratio'], logits1_ratio=cfg['model']['logits1_ratio']
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
    use_local_feature = cfg['model']['use_local_feature']

    if 'surface_loss_weight' in cfg['model']:
        surface_loss_weight = cfg['model']['surface_loss_weight']
    else:
        surface_loss_weight = 1.

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        use_local_feature=use_local_feature,
        img_size=cfg['data']['img_size'],
        surface_loss_weight=surface_loss_weight
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
    use_local_feature = cfg['model']['use_local_feature']

    if not use_local_feature:
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
    else:
        generator = generation_local.Generator3D_Local(
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


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']

    fields = {}
    fields['points'] = data.PointsField(
        cfg['data']['points_file'], points_transform,
        with_transforms=with_transforms,
        unpackbits=cfg['data']['points_unpackbits'],
    )

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
