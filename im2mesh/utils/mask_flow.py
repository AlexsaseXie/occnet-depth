import numpy as np
from im2mesh.utils.libkdtree import KDTree

def get_borders(mask):
    '''
        mask : ndarray (H * W)
        returns : ndarray (n * 2) [(x,y)]
    '''
    h, w = mask.shape
    mask_left = np.logical_not(np.concatenate((np.zeros((h, 1), dtype=bool), mask[:, :-1]), axis=1))
    mask_right = np.logical_not(np.concatenate((mask[:, 1:], np.zeros((h, 1), dtype=bool)), axis=1))
    mask_top = np.logical_not(np.concatenate((np.zeros((1, w), dtype=bool), mask[:-1, :]), axis=0))
    mask_bottom = np.logical_not(np.concatenate((mask[1:, :], np.zeros((1, w), dtype=bool)), axis=0))
    is_border = mask & (mask_left | mask_right | mask_top | mask_bottom)
    
    return np.argwhere(is_border)

def mask_flow(mask, return_borders=False):
    '''
        mask : ndarray (H * W)
        returns : ndarray (H * W) mask_flow
    '''
    h, w = mask.shape
    mask = mask.astype(bool)
    borders = get_borders(mask)
    borders_float = borders.astype(np.float32)
    # return a float 
    new_mask = mask.astype(np.float32)

    to_calculate_idx = np.nonzero(mask == 0)
    to_calculate_idx_float = np.transpose(to_calculate_idx).astype(np.float32)

    kdtree = KDTree(borders_float)
    dist, idx = kdtree.query(to_calculate_idx_float, k=1)

    new_mask[to_calculate_idx] = 1. -  dist / (np.sqrt(h * w))

    # normalize
    background = new_mask[to_calculate_idx]
    background_min = np.min(background)
    background_max = np.max(background)
    new_mask[to_calculate_idx] = (background - background_min) / (background_max - background_min)

    if return_borders:
        borders_mask = np.zeros((h,w), dtype=bool)
        borders_mask[borders[:,0], borders[:,1]] = 1.
        return new_mask, borders_mask
    else:
        return new_mask

    
    
    
        
