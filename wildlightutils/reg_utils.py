import torch

def bilateral_sem(positions, roughness_img, sem_img, albedo_img, valid_mask, sigma_albedo, sigma_pos):
    """Bilateral TV regularization with semantic weighting"""

    seg_idxs,inv_idxs,seg_counts = sem_img[sem_img != 0].unique(return_inverse=True,return_counts=True)
    ii,jj = [],[]
    # print("seg_counts", seg_counts)
    for seg_idx,seg_count in zip(seg_idxs,seg_counts):
        sample_batch = 1024
        i = torch.where(sem_img==seg_idx)[0]
        if sample_batch > seg_count:
            sample_batch = seg_count
            j = torch.arange(seg_count,device=seg_idxs.device)[None].repeat_interleave(sample_batch,0).reshape(-1)
        else:
            j = torch.randint(0,seg_count,(seg_count*sample_batch,),device=seg_idxs.device)
        # print(seg_idx, seg_count, sample_batch, j.shape)
        j = i[j]
        i = i.repeat_interleave(sample_batch,0)
        ii.append(i)
        jj.append(j)

    if len(ii) == 0:
        return torch.tensor(0.0, device=sem_img.device)
    ii = torch.cat(ii,0)
    jj = torch.cat(jj,0)

    weight_seg_ = torch.exp(-(
            (albedo_img.data[ii]-albedo_img.data[jj]).pow(2).sum(-1)
            /sigma_albedo**2)/2.0)


    weight_seg_*= torch.exp(-((positions[ii]-positions[jj]).pow(2).sum(-1)
                / sigma_pos**2)/2.0)
    weight_seg = torch.zeros(len(positions),device=positions.device)
    roughness_mean = torch.zeros(len(roughness_img),device=roughness_img.device)
    # metallic_mean = torch.zeros(len(metallic),device=metallic.device)
    # print(roughness_img[jj].shape)
    # print(weight_seg_.shape)
    roughness_mean.scatter_add_(0,ii,roughness_img[jj]*weight_seg_)
    # metallic_mean.scatter_add_(0,ii,metallic[jj].squeeze(-1)*weight_seg_)
    weight_seg.scatter_add_(0,ii,weight_seg_)
    roughness_mean = roughness_mean/(weight_seg+1e-4)
    # print("valid mask shape", valid_mask.shape)
    valid = valid_mask & (weight_seg != 0)
    # valid = weight_seg != 0
    # metallic_mean = metallic_mean/weight_seg
    # print(roughness_mean.shape, roughness_img.shape, valid.shape)
    if not valid.any():
        loss_seg_ = torch.tensor(1e-6, device=albedo_img.device, requires_grad=True) # i don't know why drjit crash if this is 0
    else:
        loss_seg_ = (roughness_mean[valid]-roughness_img[valid]).abs()

    return loss_seg_.mean()