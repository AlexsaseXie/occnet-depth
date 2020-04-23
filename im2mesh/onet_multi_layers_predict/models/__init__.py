import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.onet_multi_layers_predict.models import encoder_latent, decoder

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
    'cbatchnorm3': decoder.DecoderCBatchNorm3,
    'batchnorm_concat': decoder.DecoderBatchNormConcat,
    'concat': decoder.DecoderConcat,
}

decoder_local_dict = {
    'batchnorm_localfeature': decoder.DecoderBatchNorm_LocalFeature,
    'nobn_localfeature': decoder.Decoder_LocalFeature,
    'nobnsimple_localfeature': decoder.DecoderSimple_LocalFeature,
    'batchnormhighhidden_localfeature': decoder.DecoderBatchNormHighHidden_LocalFeature,
}


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, dataset, decoder1, decoder2, decoder3, encoder=None, encoder_latent=None, p0_z=None,
                 device=None, use_local_feature=False, logits2_ratio=1., logits1_ratio=1.):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder1 = decoder1.to(device)
        self.decoder2 = decoder2.to(device)
        self.decoder3 = decoder3.to(device)
        self.use_local_feature = use_local_feature

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None
        
        # init category_center
        category_count = len(dataset.metadata)
        self.category_count = category_count

        self._device = device
        self.p0_z = p0_z
        self.logits2_ratio = logits2_ratio
        self.logits1_ratio = logits1_ratio

    def forward(self, p, inputs, Rt=None, K=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        if self.use_local_feature:
            f3,f2,f1 = self.encode_inputs(inputs,p,Rt,K)
        else:
            f3,f2,f1 = self.encode_inputs(inputs)
        z = self.get_z_from_prior((batch_size,), sample=sample)
        p_r = self.decode(p, z, f3, f2, f1, **kwargs)
        return p_r

    def compute_elbo(self, p, occ, inputs, Rt=None, K=None, **kwargs):
        ''' Computes the expectation lower bound.

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''

        if self.use_local_feature:
            f3, f2, f1 = self.encode_inputs(inputs, p, Rt, K)
        else:
            f3, f2, f1 = self.encode_inputs(inputs)
        q_z = self.infer_z(p, occ, f3, **kwargs)
        z = q_z.rsample()
        p_r = self.decode(p, z, f3, f2, f1, **kwargs)

        rec_error = -p_r.log_prob(occ).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def encode_inputs(self, inputs, p=None, Rt=None, K=None):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            if self.use_local_feature:
                f3, f2, f1 = self.encoder(inputs, p, Rt, K)
            else:
                f3, f2, f1 = self.encoder(inputs)
        else:
            # Return inputs?
            f3 = torch.empty(inputs.size(0), 0)
            f2 = torch.empty(inputs.size(0), 0)
            f1 = torch.empty(inputs.size(0), 0)

        return f3,f2,f1

    def decode(self, p, z, f3, f2, f1, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        logits3 = self.decoder3(p, z, f3, **kwargs)
        logits2 = self.decoder2(p, z, f2, **kwargs)
        logits1 = self.decoder1(p, z, f1, **kwargs)
        logits = logits3 + self.logits2_ratio * logits2 + self.logits1_ratio * logits1 
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
