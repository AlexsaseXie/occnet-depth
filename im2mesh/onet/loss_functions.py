import torch
from torch.nn import functional as F

def get_occ_loss(logits, occ, loss_type):
    if loss_type == 'cross_entropy':
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
    elif loss_type == 'l2':
        logits = F.sigmoid(logits)
        loss_i = torch.pow((logits - occ), 2)
    elif loss_type == 'l1':
        logits = F.sigmoid(logits)
        loss_i = torch.abs(logits - occ)
    else:
        raise NotImplementedError

    return loss_i

def occ_loss_postprocess(loss_i, occ, probs, loss_tolerance_episolon=0., sign_lambda=0., threshold=0.5, surface_loss_weight=1.):
    if loss_tolerance_episolon != 0.:
        loss_i = torch.clamp(loss_i, min=loss_tolerance_episolon, max=100)
        
    if sign_lambda != 0.:
        w = 1. - sign_lambda * torch.sign(occ - 0.5) * torch.sign(probs - threshold)
        loss_i = loss_i * w

    if surface_loss_weight != 1.:
        w = ((occ > 0.) & (occ < 1.)).float()
        w = w * (surface_loss_weight - 1) + 1
        loss_i = loss_i * w

    return loss_i

def get_sdf_loss(logits, sdf, loss_type, ratio=10.):
    sdf = sdf * ratio
    if loss_type == 'l2':
        loss_i = torch.sqrt(torch.pow((logits - sdf), 2))
    elif loss_type == 'l1':
        loss_i = torch.abs(logits - sdf)
    else:
        raise NotImplementedError

    return loss_i

def sdf_loss_postprocess(loss_i, sdf, surface_loss_weight=1., surface_flag=None, surface_band=0.1):
    if surface_loss_weight != 1.:
        if surface_flag is not None:
            w = surface_flag
        else:
            w = ((sdf < surface_band) & (sdf > -surface_band)).float()
        w = w * (surface_loss_weight - 1) + 1
        loss_i = loss_i * w

    return loss_i
