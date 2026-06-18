import drjit as dr
import mitsuba as mi
import torch


def eval_discrete_laplacian_reg(data, _=None):
    """Simple discrete laplacian regularization to encourage smooth surfaces"""

    def linear_idx(p):
        p.x = dr.clamp(p.x, 0, data.shape[0] - 1)
        p.y = dr.clamp(p.y, 0, data.shape[1] - 1)
        p.z = dr.clamp(p.z, 0, data.shape[2] - 1)
        return p.z * data.shape[1] * data.shape[0] + p.y * data.shape[0] + p.x

    shape = data.shape
    z, y, x = dr.meshgrid(*[dr.arange(mi.Float, shape[i]) for i in range(3)], indexing='ij')
    p = mi.Point3i(x, y, z)
    c = dr.gather(mi.Float, data.array, linear_idx(p))
    vx0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(-1, 0, 0)))
    vx1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(1, 0, 0)))
    vy0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, -1, 0)))
    vy1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 1, 0)))
    vz0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 0, -1)))
    vz1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 0, 1)))
    laplacian = dr.sqr(c - (vx0 + vx1 + vy0 + vy1 + vz0 + vz1) / 6)
    return dr.sum(laplacian)

@dr.wrap_ad(source='drjit', target='torch')
def bilateral_sem_(roughness_img, sem_img, albedo_img, valid_img, sigma_albedo, sigma_pos):
    """Bilateral TV regularization with semantic weighting"""

    positions_y = torch.arange(sem_img.shape[0],device=sem_img.device)
    positions_x = torch.arange(sem_img.shape[1],device=sem_img.device)
    positions = torch.stack(torch.meshgrid(positions_y,positions_x),-1) # HxWx2
    unit_size = torch.tensor(sem_img.shape,device=sem_img.device)
    positions = positions / unit_size
    positions = positions.reshape(-1,2)
    orig_roughness_img_shape = roughness_img.shape
    roughness_img = roughness_img[:, :, 0].reshape(-1)
    sem_img = sem_img.reshape(-1)
    albedo_img = albedo_img.reshape(-1,3)
    valid_mask = valid_img == 1
    valid_mask = valid_mask.reshape(-1)

    seg_idxs,inv_idxs,seg_counts = sem_img[sem_img != 0].unique(return_inverse=True,return_counts=True)
    ii,jj = [],[]
    for seg_idx,seg_count in zip(seg_idxs,seg_counts):
        sample_batch = 1024
        i = torch.where(sem_img==seg_idx)[0]
        if sample_batch > seg_count:
            sample_batch = seg_count
            j = torch.arange(seg_count,device=seg_idxs.device)[None].repeat_interleave(sample_batch,0).reshape(-1)
        else:
            j = torch.randint(0,seg_count,(seg_count*sample_batch,),device=seg_idxs.device)
        print(seg_idx, seg_count, sample_batch, j.shape)
        j = i[j]
        i = i.repeat_interleave(sample_batch,0)
        ii.append(i)
        jj.append(j)

    roughness_vis_img =  torch.zeros(orig_roughness_img_shape[:2], device=sem_img.device)
    vaid_vis_img =  torch.zeros(orig_roughness_img_shape[:2], device=sem_img.device)
    debug_imgs = {
        "roughness_mean": roughness_vis_img,
        "valid_mask": vaid_vis_img
    }
    if len(ii) == 0:
        return torch.tensor(0.0, device=sem_img.device),debug_imgs
    ii = torch.cat(ii,0)
    jj = torch.cat(jj,0)

    weight_seg_ = torch.exp(-(
            (albedo_img.data[ii]-albedo_img.data[jj]).pow(2).sum(-1)
            /sigma_albedo**2)/2.0)


    weight_seg_*= torch.exp(-((positions[ii]-positions[jj]).pow(2).sum(-1)
                / sigma_pos**2)/2.0)
    weight_seg = torch.zeros(len(positions),device=positions.device)
    roughness_mean = torch.zeros(len(roughness_img),device=roughness_img.device)
    roughness_mean.scatter_add_(0,ii,roughness_img[jj].squeeze(-1)*weight_seg_)
    weight_seg.scatter_add_(0,ii,weight_seg_)
    roughness_mean = roughness_mean/(weight_seg+1e-4)
    valid = valid_mask & (weight_seg != 0)
    if valid.sum() == 0:
        loss_seg_ = torch.tensor(1e-6, device=albedo_img.device, requires_grad=True) # i don't know why drjit crash if this is 0
    else:
        loss_seg_ = (roughness_mean[valid]-roughness_img.squeeze(-1)[valid]).abs()
    roughness_vis_img.view(-1)[:] = roughness_mean
    vaid_vis_img.view(-1)[:] = valid
    vaid_vis_img.requires_grad = True
    return loss_seg_, debug_imgs

def bilateral_sem(roughness_img, sem_img, albedo_img, valid_img, sigma_albedo, sigma_pos):
    """Bilateral TV regularization with semantic weighting"""
    loss_seg, debug_imgs = bilateral_sem_(roughness_img, sem_img, albedo_img, valid_img, sigma_albedo, sigma_pos)
    dr.make_opaque(loss_seg)
    dr.make_opaque(debug_imgs)
    return dr.select(loss_seg.array==0.0, mi.Float(0.0), loss_seg.array), debug_imgs
