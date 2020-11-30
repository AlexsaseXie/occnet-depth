import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.onet_depth.models.space_carver import *
from im2mesh.onet.models import encoder_latent, decoder as decoder2
from im2mesh.onet_multi_layers_predict.models import decoder as decoder1
from im2mesh.onet.loss_functions import get_occ_loss, occ_loss_postprocess
from im2mesh.encoder.pointnet import feature_transform_reguliarzer

# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
}

# Decoder dictionary
decoder_dict = {
    'simple': decoder2.Decoder,
    'cbatchnorm': decoder2.DecoderCBatchNorm,
    'cbatchnorm2': decoder2.DecoderCBatchNorm2,
    'batchnorm': decoder2.DecoderBatchNorm,
    'cbatchnorm_noresnet': decoder2.DecoderCBatchNormNoResnet,
}

decoder_local_dict = {
    'batchnorm_localfeature': decoder1.DecoderBatchNorm_LocalFeature,
    'batchnormsimple_localfeature': decoder1.DecoderBatchNormSimple_LocalFeature,
    'nobn_localfeature': decoder1.Decoder_LocalFeature,
    'nobnsimple_localfeature': decoder1.DecoderSimple_LocalFeature,
    'batchnormhighhidden_localfeature': decoder1.DecoderBatchNormHighHidden_LocalFeature,
}


def background_setting(depth_maps, gt_masks, v=0.):
    #inplace function
    depth_maps[1. - gt_masks] = v

class OccupancyWithDepthNetwork(nn.Module):
    ''' OccupancyWithDepth Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network( eithor for depth map or for depth pointcloud )
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, depth_predictor=None, decoder=None, encoder=None, encoder_latent=None, p0_z=None,
                 device=None, decoder_local=None, local_logit_ratio=1., space_carver_mode=None, space_carver_eps=None,
                 space_carver_drop_p=None):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        if depth_predictor is not None:
            self.depth_predictor = depth_predictor
        else:
            self.depth_predictor = None
        
        if decoder is not None: 
            self.decoder = decoder
        else:
            self.decoder = None

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent
        else:
            self.encoder_latent = None

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

        if decoder_local is not None:
            self.decoder_local = decoder_local
        else:
            self.decoder_local = None

        self.space_carver_mode = space_carver_mode
        if self.space_carver_mode:
            additional_params = {}
            if space_carver_eps is not None:
                additional_params['eps'] = space_carver_eps
            if space_carver_drop_p is not None:
                additional_params['training_drop_carving_p'] = space_carver_drop_p

            self.space_carver = SpaceCarverModule(mode=self.space_carver_mode, **additional_params)

        self._device = device
        self.p0_z = p0_z
        self.local_logit_ratio = local_logit_ratio

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

    def forward(self, p, inputs, gt_mask=None, sample=True, 
        func='forward', halfway=False,
        # following params are for forward func 
        train_loss=False, return_depth_map=False,
        occ=None, loss_type='cross_entropy', loss_tolerance_episolon=0., 
        sign_lambda=0., threshold=0.5, surface_loss_weight=1., **kwargs):
        ''' Performs a forward pass through the network.

        its own function only predicts depth map and encodes
        not supporting decoder_local

        new: made to be the entrance for all functions to support DP
        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
            halfway: whether to forward from intermediate inputs
        '''

        # default forward: forward_full
        assert func in ('forward', 'compute_elbo', 
            'encode', 'encode_first_step', 'encode_second_step', 
            'decode')

        if func == 'compute_elbo':
            assert occ is not None
            if halfway == False:
                return self.compute_elbo(p, occ, inputs, gt_mask=gt_mask, **kwargs)
            else:
                return self.compute_elbo_halfway(p, occ, inputs, gt_mask=gt_mask, **kwargs)
        elif func == 'forward':
            if halfway:
                return self.forward_halfway(p, inputs, sample=sample, train_loss=train_loss, 
                            occ=occ, loss_type=loss_type, loss_tolerance_episolon=loss_tolerance_episolon, 
                            sign_lambda=sign_lambda, threshold=threshold, surface_loss_weight=surface_loss_weight,
                            **kwargs)
            else:
                return self.forward_full(p, inputs, sample=sample, train_loss=train_loss, 
                            occ=occ, loss_type=loss_type, loss_tolerance_episolon=loss_tolerance_episolon, 
                            sign_lambda=sign_lambda, threshold=threshold, surface_loss_weight=surface_loss_weight,
                            **kwargs)
        else:
            raise NotImplementedError

    def forward_full(self, p, inputs, gt_mask=None, sample=True, 
            train_loss=False, return_depth_map=False,
            occ=None, loss_type='cross_entropy', loss_tolerance_episolon=0., 
            sign_lambda=0., threshold=0.5, surface_loss_weight=1., **kwargs):
        ### here begins its own function
        if train_loss:
            assert occ is not None

        batch_size = p.size(0)
        depth_map = self.predict_depth_map(inputs)
        background_setting(depth_map, gt_mask) 
        c = self.encode(depth_map)

        if train_loss:
            q_z = self.infer_z(p, occ, c, **kwargs)
            z = q_z.rsample()
        else:
            z = self.get_z_from_prior((batch_size,), sample=sample)
        
        p_r = self.decode(p, z, c, **kwargs)

        if train_loss:
            loss = self.calc_loss(p_r, occ, q_z=q_z, trans_feature=None, loss_type=loss_type,
                loss_tolerance_episolon=loss_tolerance_episolon, sign_lambda=sign_lambda, 
                threshold=threshold, surface_loss_weight=surface_loss_weight)

            if not return_depth_map:
                return loss, p_r
            else:
                return loss, depth_map, p_r
        else:
            if not return_depth_map:
                return p_r
            else:
                return depth_map, p_r

    def forward_halfway(self, p, encoder_input, sample=True, train_loss=False, 
        occ=None, loss_type='cross_entropy', loss_tolerance_episolon=0., 
        sign_lambda=0., threshold=0.5, surface_loss_weight=1., **kwargs):

        batch_size = p.size(0)
        if train_loss:
            assert occ is not None
            c = self.encode(encoder_input, only_feature=False, p=p)

            if isinstance(c, tuple):
                trans_feature = c[-1]
                if self.decoder_local is None:
                    c = c[0]
                else:
                    c = (c[0], c[1])
            else:
                trans_feature = None
        else:
            c = self.encode(encoder_input, only_feature=True, p=p)

        if train_loss:
            q_z = self.infer_z(p, occ, c, **kwargs)
            z = q_z.rsample()
        else:
            z = self.get_z_from_prior((batch_size,), sample=sample)

        # warning: **kwargs is used in space carver
        p_r = self.decode(p, z, c, **kwargs)

        if train_loss:
            loss = self.calc_loss(p_r, occ, q_z=q_z, trans_feature=trans_feature, loss_type=loss_type,
                loss_tolerance_episolon=loss_tolerance_episolon, sign_lambda=sign_lambda, 
                threshold=threshold, surface_loss_weight=surface_loss_weight)

            return loss, p_r
        else:
            return p_r

    def compute_elbo(self, p, occ, inputs, gt_mask, **kwargs):
        ''' Computes the expectation lower bound.

        its own function only predicts depth map and encodes
        not supporting decoder_local

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''
        depth_map = self.predict_depth_map(inputs)
        background_setting(depth_map, gt_mask)
        c = self.encode(depth_map)
        q_z = self.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()
        p_r = self.decode(p, z, c, **kwargs)

        rec_error = -p_r.log_prob(occ).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def compute_elbo_halfway(self, p, occ, encoder_input, **kwargs):
        c = self.encode(encoder_input, p=p)
        q_z = self.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()
        p_r = self.decode(p, z, c, **kwargs)

        rec_error = -p_r.log_prob(occ).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def encode(self, encoder_input, only_feature=True, p=None):
        ''' Encodes the depth map / depth pointcloud.

        Args:
             encoder_input (tensor) or (dict of tensors): depth map / depth pointcloud
        '''
        assert self.encoder is not None 

        if self.decoder_local is None:
            c = self.encoder(encoder_input)
        else:
            assert p is not None
            c = self.encoder.forward_local(encoder_input, p)

        if only_feature and isinstance(c, tuple):
            if self.decoder_local is None:
                c = c[0]
            else:
                c = (c[0], c[1])

        return c

    def decode(self, p, z, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        if self.space_carver_mode:
            assert 'reference' in kwargs
            space_carver_kwargs = { 'reference': kwargs['reference']}
            if 'cor_occ' in kwargs:
                space_carver_kwargs['cor_occ'] = kwargs['cor_occ']
            if 'world_mat' in kwargs:
                space_carver_kwargs['world_mat'] = kwargs['world_mat']
            if 'camera_mat' in kwargs:
                space_carver_kwargs['camera_mat'] = kwargs['camera_mat']

            remove_idx_bool = self.space_carver(p, **space_carver_kwargs)

        if self.decoder_local is None:
            logits = self.decoder(p, z, c, **kwargs)
        else:
            logits_global = self.decoder(p, z, c[0], **kwargs)
            logits_local = self.decoder_local(p, z, c[1], **kwargs)
            logits = logits_global + self.local_logit_ratio * logits_local

        if self.space_carver_mode and not self.training:
            # eval phase behaviour
            # give very large negative value
            logits[remove_idx_bool] = -50.

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

    def calc_loss(self, p_r, occ, q_z=None, trans_feature=None, loss_type='cross_entropy',
        loss_tolerance_episolon=0., sign_lambda=0., threshold=0.5, surface_loss_weight=1.):
        logits = p_r.logits
        probs = p_r.probs

        loss = 0

        if q_z is not None:
            kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
            loss = loss + kl.mean()

        if trans_feature is not None:
            loss = loss + 0.001 * feature_transform_reguliarzer(trans_feature) 

        loss_i = get_occ_loss(logits, occ, loss_type=loss_type)

        loss_i = occ_loss_postprocess(loss_i, occ, probs, 
            loss_tolerance_episolon=loss_tolerance_episolon, 
            sign_lambda=sign_lambda, threshold=threshold, 
            surface_loss_weight=surface_loss_weight
        )

        loss = loss + loss_i.sum(-1).mean()
        return loss 
