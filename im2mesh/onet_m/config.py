import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.onet_m import models, training, generation
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
        dataset, decoder, encoder, encoder_latent, p0_z, device=device
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

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        calc_feature_category_loss=False,
        record_feature_category=False, 
        attractive_p=cfg['training']['attractive_p'], 
        repulsive_p=cfg['training']['repulsive_p'],
        feature_k=cfg['training']['feature_k']
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


from im2mesh.onet.config import get_prior_z 
    


from im2mesh.onet.config import get_data_fields