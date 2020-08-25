import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.onet_depth.models import encoder_latent, decoder

# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
}

# Decoder dictionary
decoder_dict = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
    'cbatchnorm2': decoder.DecoderCBatchNorm2,
    'batchnorm': decoder.DecoderBatchNorm,
    'cbatchnorm_noresnet': decoder.DecoderCBatchNormNoResnet,
}


def background_setting(depth_maps, gt_masks, v=0.):
    #inplace function
    depth_maps[1. - gt_masks] = v

class OccupancyWithDepthNetwork(nn.Module):
    ''' OccupancyWithDepth Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, depth_predictor=None, decoder=None, encoder=None, encoder_latent=None, p0_z=None,
                 device=None):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        if depth_predictor is not None:
            self.depth_predictor = depth_predictor.to(device)
        else:
            self.depth_predictor = None
        
        if decoder is not None: 
            self.decoder = decoder.to(device)
        else:
            self.decoder = None

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device
        self.p0_z = p0_z

    def predict_depth_maps(self, inputs):
        #batch_size = inputs.size(0)
        assert self.depth_predictor is not None
        return self.depth_predictor(inputs)

    def predict_depth_map(self, inputs):
        #batch_size = inputs.size(0)
        assert self.depth_predictor is not None
        return self.depth_predictor.get_last_predict(inputs)

    def fetch_minmax(self):
        assert self.depth_predictor is not None
        return self.depth_predictor.fetch_minmax()

    def forward(self, p, inputs, gt_mask, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        depth_map = self.predict_depth_map(inputs)
        background_setting(depth_map, gt_mask)       
        c = self.encode_depth_map(depth_map)
        z = self.get_z_from_prior((batch_size,), sample=sample)
        p_r = self.decode(p, z, c, **kwargs)
        return p_r

    def forward_halfway(self, p, depth_map, sample=True, **kwargs):
        batch_size = p.size(0)   
        c = self.encode_depth_map(depth_map)
        z = self.get_z_from_prior((batch_size,), sample=sample)
        p_r = self.decode(p, z, c, **kwargs)
        return p_r

    def compute_elbo(self, p, occ, inputs, gt_mask, **kwargs):
        ''' Computes the expectation lower bound.

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''
        depth_map = self.predict_depth_map(inputs)
        background_setting(depth_map, gt_mask)
        c = self.encode_depth_map(depth_map)
        q_z = self.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()
        p_r = self.decode(p, z, c, **kwargs)

        rec_error = -p_r.log_prob(occ).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def compute_elbo_halfway(self, p, occ, depth_map, **kwargs):
        c = self.encode_depth_map(depth_map)
        q_z = self.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()
        p_r = self.decode(p, z, c, **kwargs)

        rec_error = -p_r.log_prob(occ).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def encode_depth_map(self, depth_map):
        ''' Encodes the depth map.

        Args:
             depth_map (tensor): the depth map
        '''

        if self.encoder is not None:
            c = self.encoder(depth_map)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, z, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, z, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, p, occ, c, **kwargs):
        ''' Infers z.

        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
        else:
            z = self.p0_z.mean.to(self._device)
            z = z.expand(*size, *z.size())

        return z

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
