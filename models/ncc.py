import torch
import torch.nn.functional as  F
import math
def compute_ncc(p1, p2): #p1: (BxCxL)
    batch_size, n_channel, patch_size = p1.shape
    p1_p2 = p1 * p2
    
    p1_sq = p1.pow(2)
    p2_sq = p2.pow(2)
    
    p1_sum = p1.sum(dim=-1)
    p2_sum = p2.sum(dim=-1)
    p1_p2_sum = p1_p2.sum(dim=-1)
    
    p1_sq_sum = p1_sq.sum(dim=-1)
    p2_sq_sum = p2_sq.sum(dim=-1)

    u_p1 = p1_sum / patch_size
    u_p2 = p2_sum / patch_size
    # print(p1_p2_sum, u_p1 * p2_sum,  u_p2*p1_sum , u_p1*u_p2*patch_size)
    cross = p1_p2_sum - u_p1 * p2_sum - u_p2*p1_sum + u_p1*u_p2*patch_size
    p1_var = p1_sq_sum - 2 * u_p1 * p1_sum + u_p1 * u_p1 * patch_size
    p2_var = p2_sq_sum - 2 * u_p2 * p2_sum + u_p2 * u_p2 * patch_size
    # print('cross', cross)
    # print('p1_var', p1_var)
    # print('p2_var', p2_var)
    cc = cross * cross / (p1_var * p2_var + 1e-5)
    # print('cc', cc)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0) #BxC 
    
    return ncc, p2_var

def loss_ncc(color, gt): #Bx3
    patch_size = int(math.sqrt(color.shape[0]))
    # print("color_here", color.shape, "gt_here", gt.shape)
    color = color.reshape(patch_size, patch_size,3).unsqueeze(dim=0).permute(0, 3, 1, 2) #1x3xPxP
    gt = gt.reshape(patch_size, patch_size,3).unsqueeze(dim=0).permute(0, 3, 1, 2) #1x3xPxP
    # print("color", color.shape)
    # print("gt", gt.shape)
    # print(F.unfold(color, kernel_size=11).shape)
    WIN_SIZE = 22
    color = F.unfold(color, kernel_size=WIN_SIZE).reshape(3, WIN_SIZE*WIN_SIZE, -1).permute(2, 0, 1) #1x3*11*11x144?
    gt = F.unfold(gt, kernel_size=WIN_SIZE).reshape(3, WIN_SIZE*WIN_SIZE, -1).permute(2,0,1)
    # print(color.shape)
    # print(gt.shape)
    color = color.mean(dim=1, keepdim=True) #144x1xpatch_size
    gt = gt.mean(dim=1, keepdim=True) #144x1xpatch_size
    # print(color.shape)
    # print(gt.shape)
    ncc, p2_var = compute_ncc(color, gt) #144 x 1
    # print("ncc", ncc)
    return ncc.mean(axis=-1), p2_var

