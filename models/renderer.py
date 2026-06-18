import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
from models.physicalshader import unpack_brdf_params_burley
import time
import nerfstudio.exporter.marching_cubes

def extract_fields(bound_min, bound_max, resolution, query_func, device):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, device):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func, device=device)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    device = bins.device
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def locate_intersection(sdfs, z_vals, mid_z_vals, rays_o, rays_d): # From geo-neus
    batch_size, n_samples = z_vals.shape
    
    sdf_d = sdfs.reshape(batch_size, n_samples)
    prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
    sign = prev_sdf * next_sdf
    sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
    idx = reversed(torch.Tensor(range(1, n_samples)).cuda())
    tmp = torch.einsum("ab,b->ab", (sign, idx))
    prev_idx = torch.argmax(tmp, 1, keepdim=True)
    next_idx = prev_idx + 1

    sdf1 = torch.gather(sdf_d, 1, prev_idx)
    sdf2 = torch.gather(sdf_d, 1, next_idx)
    z_vals1 = torch.gather(mid_z_vals, 1, prev_idx)
    z_vals2 = torch.gather(mid_z_vals, 1, next_idx)

    z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
    z_vals_sdf0 = torch.where(z_vals_sdf0 < 0, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
    max_z_val = torch.max(z_vals)
    
    z_vals_sdf0 = torch.where(z_vals_sdf0 > max_z_val, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
    pts_sdf0 = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_sdf0[..., :, None]  # [batch_size, 1, 3]
    
    return z_vals_sdf0, pts_sdf0

class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                #  brdf_settings,
                 perturb,
                 need_hess):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.need_hess = need_hess
        # self.options = options
        # if color_network.trichromatic:
        #     self.n_specular_channels = 3
        # else:
        # self.n_specular_channels = 1
        # self.n_brdf_dim = color_network.n_brdf_dim
        if hasattr(color_network, "n_brdf_dim"):
            self.n_brdf_dim = color_network.n_brdf_dim
        else:
            self.n_brdf_dim = 0
        self.perturb = perturb




    def render_core_outside(self, rays_o, rays_d, light_o, light_lum, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs, light_o, light_lum)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        # print("outside color", sampled_color.min(), sampled_color.max(), sampled_color.mean())
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }


    def render_core_outside_alpha(self, rays_o, rays_d, light_o, light_lum, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, _ = nerf(pts, dirs, light_o, light_lum, True)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        

        return {
            'alpha': alpha,
            'weights': weights,
        }
    
    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        device = rays_o.device
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        device = rays_o.device
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    @torch.no_grad()
    def extract_shading_params(self, pts):
        '''
        pts: (..., 3)
        '''
        pts = torch.from_numpy(pts).float().cuda()

        sdf_nn_output = self.sdf_network(pts)
        normals = torch.nn.functional.normalize(self.sdf_network.gradient(pts), dim=-1)
        n_brdf_params = self.n_brdf_dim
        sd = sdf_nn_output[:, 0:1]
        brdf_params = sdf_nn_output[:, 1:1+n_brdf_params]

        subsurface, metallic, specular, clearcoat, roughness, clearcoat_gloss, base_color = unpack_brdf_params_burley(
                            brdf_params, 
                            self.color_network.bsdf_config
                        )
        # return {
        #     'diffuse_albedo': diffuse_albedo.detach().cpu().numpy(), #(..., 3)
        #     'specular_albedo': specular_albedo.detach().cpu().numpy() * [1.0,1.0,1.0], #(n_lobes, ..., 3)
        #     'roughness': roughness.detach().cpu().numpy(), #(n_lobes, ..., 1)
        #     'r0': r0.detach().cpu().numpy(), #(n_lobes, ..., 1)
        # }

        return dict(
            signed_distance = sd.detach().cpu().numpy(),
            object_normal = normals.detach().cpu().numpy(),
            subsurface = subsurface.detach().cpu().numpy(),
            metallic = metallic.detach().cpu().numpy(),
            specular = specular.detach().cpu().numpy(),
            clearcoat = clearcoat.detach().cpu().numpy(),
            roughness = roughness.detach().cpu().numpy(),
            clearcoat_gloss = clearcoat_gloss.detach().cpu().numpy(),
            base_color = base_color.detach().cpu().numpy(),
        )
        
    @torch.no_grad()
    def extract_shading_diffuse(self, pts):
        '''
        pts: (..., 3)
        '''
        pts = torch.from_numpy(pts).float().cuda()

        sdf_nn_output = self.sdf_network(pts)
        normals = torch.nn.functional.normalize(self.sdf_network.gradient(pts), dim=-1)
        n_brdf_params = self.n_brdf_dim
        sd = sdf_nn_output[:, 0:1]
        brdf_params = sdf_nn_output[:, 1:1+n_brdf_params]

        base_color = brdf_params
        # return {
        #     'diffuse_albedo': diffuse_albedo.detach().cpu().numpy(), #(..., 3)
        #     'specular_albedo': specular_albedo.detach().cpu().numpy() * [1.0,1.0,1.0], #(n_lobes, ..., 3)
        #     'roughness': roughness.detach().cpu().numpy(), #(n_lobes, ..., 1)
        #     'r0': r0.detach().cpu().numpy(), #(n_lobes, ..., 1)
        # }

        return dict(
            signed_distance = sd.detach().cpu().numpy(),
            object_normal = normals.detach().cpu().numpy(),
            base_color = base_color.detach().cpu().numpy(),
        )
        

    def render_core(self,
                    rays_o,
                    rays_d,
                    light_o,
                    light_lum,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        device = rays_o.device
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([sample_dist], device=device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        # input("before sdf call")
        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        sdf.requires_grad_(True)
        # if sdf.requires_grad:
        #     sdf.retain_grad()
        brdf_params = sdf_nn_output[:, 1:1+self.n_brdf_dim]
        feature_vector = sdf_nn_output[:, 1+self.n_brdf_dim:]
        # input("before grad call")
        if not self.need_hess:
            gradients = sdf_network.gradient(pts).squeeze()
        else:
            _, gradients, hess = sdf_network.eval_all(pts)
        gradients.requires_grad_(True)
        # print("gradients", gradients.dtype)
        # input("after grad call")
        # print("disable color network parameters grad")
        # for name, parameter in color_network.named_parameters():
        #     parameter.requires_grad = False
        sampled_color, extra_out = color_network(pts, gradients, dirs, light_o, light_lum, brdf_params, feature_vector)
        # print("restore color network parameters grad")
        # for name, parameter in color_network.named_parameters():
        #     parameter.requires_grad = True
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        # print("WARNING detaching sampled color just to see how effective radiosity loss is")
        # sampled_color = sampled_color.detach()
        # input("color net call")
        #sampled_color = gradients / torch.norm(gradients, p=2, dim=-1, keepdim=True) * 0.5 + 0.5
        #sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        # print("sampled_color", sampled_color.min(), sampled_color.max(), sampled_color.mean())
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        # inv_s = inv_s * 0 + 100000
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        trace_grad_implementation = False
        if trace_grad_implementation: # this is just wrong. fix it if you have time
            assert background_alpha is None
            inv_s = inv_s.expand(batch_size * n_samples, n_samples, 1)
            inv_s_orig = inv_s
            true_cos = (dirs * gradients).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf.unsqueeze(dim=1) * inv_s) # B*s x copied x 1 (x)
            next_cdf = torch.sigmoid(estimated_next_sdf.unsqueeze(dim=1) * inv_s) # B*s x Copied x 1 (x-1)
            # print("prev_cdf next_cdf", prev_cdf.shape, next_cdf.shape)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples, n_samples).clip(0.0, 1.0) # b x s x copied

            T = torch.cumprod(torch.cat([torch.ones([batch_size, 1, n_samples], device=device), 1. - alpha + 1e-7], 1), 1)[:, :-1, :] #B x N x N # first row: no inv s. second row: inv_s0,0, inv_s0,1; inv_s0,2; third row: invs0,0*invs1,0; inv_s0,1*inv_s1,1; inv_s0,2*inv_s1,2
            # T2 = torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha[:, :, 0] + 1e-7], -1), -1)[:, :-1] #B x N x N
            # print(alpha.shape, T.shape,  torch.diagonal(T, dim1=-2, dim2=-1).shape)
            # weights = alpha[:, :, 0] * torch.diagonal(T, dim1=-2, dim2=-1)
            # print("T", T, T.shape)
            # print("diag", torch.diagonal(T, dim1=-2, dim2=-1), torch.diagonal(T, dim1=-2, dim2=-1).shape)
            # print("diags", torch.diag(T[0]))
            # print("first diag",  torch.diagonal(T, dim1=-2, dim2=-1)[0])
            weights = torch.diagonal(alpha, dim1=-2, dim2=-1) * torch.diagonal(T, dim1=-2, dim2=-1) # first row: inv_s0,0 second row: inv_s1,1 * inv_s0,1 third row: inv_s2,2 * inv_s0,2 * inv_s1,2
            # print(T)
            # print(torch.diagonal(T, dim1=-2, dim2=-1))
            # print("T", T.shape, torch.diagonal(T, dim1=-2, dim2=-1).shape)

            c = c[:, 0, :]
            inv_s = inv_s[:, 0, :]
        else:
            inv_s = inv_s.expand(batch_size * n_samples, 1)
            inv_s_orig = inv_s
            true_cos = (dirs * gradients).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)


            # Render with background
            if background_alpha is not None:
                alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
                alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
                test_bg = background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
                # print(test_bg.min(), test_bg.max(), test_bg.mean())
                # print(background_sampled_color[:, n_samples:].min(), background_sampled_color[:, n_samples:].max(), background_sampled_color[:, n_samples:].mean())
                sampled_color = sampled_color * inside_sphere[:, :, None] +\
                                background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
                sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)
                test_back_ground_alpha = background_alpha[:, :n_samples] * (1.0 - inside_sphere)
                # print("test_background_alpha", test_back_ground_alpha.min(),test_back_ground_alpha.max(), test_back_ground_alpha.mean())
                # print("background_alpha_outside", background_alpha[:, n_samples:].min(),background_alpha[:, n_samples:].max(), background_alpha[:, n_samples:].mean())

                
            # print("WARNING!!!! using outside as color")
            # sampled_color[:, :n_samples] = sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            # alpha[:, :n_samples] = alpha[:, :n_samples] * (1.0 - inside_sphere)
            # alpha[:, :n_samples][(sampled_color[:, :n_samples].mean(dim=-1)<0.1) & (inside_sphere!=0.0)] = 0
            weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        if background_alpha is not None:
            test_weights_bg = weights[:, :n_samples] * (1.0 - inside_sphere)
            # print("test_weights_bg", test_weights_bg.min(), test_weights_bg.max(), test_weights_bg.mean())
            test_weights_bg_outside = weights[:, n_samples:]
            # print("test_weights_bg_outside", test_weights_bg_outside.min(), test_weights_bg_outside.max(), test_weights_bg_outside.mean())

        weights_sum = weights.sum(dim=-1, keepdim=True)
        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        # print("color", color.min(), color.max(), color.mean())
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)
        # back facing penalty
        # per_grid_back_facing_penalty = (torch.clamp((F.normalize(dirs, dim=-1) * F.normalize(gradients, dim=-1)).sum(dim=-1), min=0.0)**2).reshape(batch_size, n_samples)
        # print("sampled_color", sampled_color.shape, weights.shape, per_grid_back_facing_penalty.shape)
        # back_facing_loss = (per_grid_back_facing_penalty * weights).sum()
        
        # back facing penalty
        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        if self.need_hess:
            hess_error = (relax_inside_sphere[:, :, None] * hess.reshape(batch_size, n_samples, 9).abs()).sum() / (relax_inside_sphere.sum() + 1e-5)
        
        
        # if 'light_net_color' in extra_out:
        #     # extra_out['light_net_color'] = (extra_out['light_net_color'].reshape(batch_size, n_samples, 3)).mean(dim=1)
        #     extra_out['light_net_color'] = (extra_out['light_net_color'].reshape(batch_size, n_samples, 3) * weights[:, :, None].detach()).sum(dim=1)
        # with torch.no_grad():
        #     pts_intersect = locate_intersection(sdf, z_vals, mid_z_vals, rays_o, rays_d).reshape(-1, 3)
        #     # intersect_gradient = sdf_network.gradient(pts_intersect).squeeze()
        # pts_intersect = locate_intersection(sdf, z_vals, mid_z_vals, rays_o, rays_d)#.reshape(-1, 3)
        result= {
            'color': color,
            'sdf': sdf.reshape(batch_size, n_samples),
            "sdf_grad": sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            'gradients_orig': gradients,
            # 'back_facing_loss': back_facing_loss,
            's_val': 1.0 / inv_s,
            'inv_s': inv_s_orig,
            'mid_z_vals': mid_z_vals,
            'z_vals': z_vals,
            # 'rays_o': rays_o,
            # 'rays_d': rays_d,
            'weights': weights,
            'brdf_params': brdf_params.reshape(batch_size, n_samples, brdf_params.shape[-1]),
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'z': mid_z_vals,
            'extra_out': extra_out,
            'sampled_color': sampled_color.reshape(batch_size, n_samples, sampled_color.shape[-1]),
            'feature_vector': feature_vector.reshape(batch_size, n_samples, feature_vector.shape[-1]),
            # 'sdf': sdf.reshape(batch_size, n_samples),
            # "pts_intersect": pts_intersect,
            # "intersect_gradient": intersect_gradient
            
        }
        if self.need_hess:
            result["hess_error"] = hess_error
        return result

    def render_core_alpha(self,
                    rays_o,
                    rays_d,
                    light_o,
                    light_lum,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape
        device = rays_o.device
        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([sample_dist], device=device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        brdf_params = sdf_nn_output[:, 1:1+self.n_brdf_dim]
        feature_vector = sdf_nn_output[:, 1+self.n_brdf_dim:]

        gradients = sdf_network.gradient(pts).squeeze()
        #sampled_color = gradients / torch.norm(gradients, p=2, dim=-1, keepdim=True) * 0.5 + 0.5
        #sampled_color = sampled_color.reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        # inv_s = inv_s * 0 + 100000
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        inv_s.requires_grad_(True)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        # print("true_cos", true_cos.reshape(batch_size, n_samples))

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        # print("iter cos before", iter_cos)
        # iter_cos_orig = iter_cos
        # iter_cos = -(F.relu(-iter_cos  - 1.0) * (1.0 - neg_cos_anneal_ratio) +
        #             F.relu(-iter_cos) * neg_cos_anneal_ratio)  # always non-positive
        # print("iter cos after", iter_cos)
        # print("cos_anneal_ratio", cos_anneal_ratio)
        # print("iter_cos", iter_cos.reshape(batch_size, n_samples))
        # print("iter_cos", iter_cos, cos_anneal_ratio, true_cos)
        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        # print("alpha", alpha)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'sdf': sdf,
            'dists': dists.reshape(batch_size, n_samples),
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }
    def sample_z(self, rays_o, rays_d, near, far, perturb_overwrite=-1):
        t_init = time.time()
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(rays_o.device)
        z_vals = near + (far - near) * z_vals[None, :]
        device = rays_o.device
        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        z_vals = z_vals.clip(min=1e-3)
        #print("z_vals", z_vals)
        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance
        return batch_size, n_samples, z_vals, z_vals_outside, sample_dist
    def render(self, rays_o, rays_d, light_o, light_lum, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        # print("rays_o here", rays_o.shape)

        background_alpha = None
        background_sampled_color = None

        batch_size, n_samples, z_vals, z_vals_outside, sample_dist = self.sample_z(rays_o, rays_d, near, far, perturb_overwrite)
        #print("z_vals after is", z_vals)
        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, light_o, light_lum, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']
        t_imp_sample = time.time()
        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    light_o,
                                    light_lum, 
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)
        t_rndr_core = time.time()
        # print("t_importance", t_imp_sample - t_init, "t_rndr_core", t_rndr_core - t_imp_sample)
        
        # color_fine,
        # locate_intersection(sdf, z_vals, mid_z_vals, rays_o, rays_d)#.reshape(-1, 3)
        result = {
            'color_fine':  color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'gradients_orig': ret_fine['gradients_orig'],
            'weights': weights,
            'brdf_params': ret_fine['brdf_params'],
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'z': ret_fine['z'],
            # 'back_facing_loss': ret_fine['back_facing_loss']
            'extra_out': ret_fine['extra_out'],
            'sampled_color': ret_fine['sampled_color'],
            'inside_sphere': ret_fine['inside_sphere'],
            # 'pts_intersect': ret_fine['pts_intersect'],
            # 'intersect_gradient': ret_fine['intersect_gradient'],
            'feature_vector': ret_fine['feature_vector'],
            'sdf': ret_fine['sdf'],
            "sdf_grad": ret_fine["sdf_grad"],
            'inv_s': ret_fine['inv_s'],
            'dists': ret_fine['dists'],
            'z_vals': ret_fine['z_vals'],
            'mid_z_vals': ret_fine['mid_z_vals'],
            # 'rays_o': ret_fine['rays_o'],
            # 'rays_d': ret_fine['rays_d']
        }
        if 'hess_error' in ret_fine:
            result['hess_error'] = ret_fine['hess_error']
        
        return result
    def render_alpha(self, rays_o, rays_d, light_o, light_lum, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        device = rays_o.device
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device=device)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]], device=device)
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        z_vals = z_vals.clip(min=1e-3)

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside_alpha(rays_o, rays_d, light_o, light_lum, z_vals_feed, sample_dist, self.nerf)

            background_alpha = ret_outside['alpha']
        # print("render_core_alpha")
        # Render core
        ret_fine = self.render_core_alpha(rays_o,
                                    rays_d,
                                    light_o,
                                    light_lum, 
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=None,
                                    cos_anneal_ratio=cos_anneal_ratio)

        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, device, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                device=device,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))

    def extract_geometry_hires(self, bound_min, bound_max, resolution, threshold=0.0):
        def get_sdf(pts):
            N = 10000
            pts_all_batch = torch.split(pts, N)
            sdfs = []
            for pts_batch in pts_all_batch:
                sdf = self.sdf_network.sdf(pts_batch).squeeze(dim=-1)
                sdfs.append(sdf)
            return torch.cat(sdfs, dim=0)
        mesh = nerfstudio.exporter.marching_cubes.generate_mesh_with_multires_marching_cubes(
            get_sdf,
            resolution=resolution,
            bounding_box_min=bound_min,
            bounding_box_max=bound_max,
            isosurface_threshold=threshold
        )
        verts = mesh.vertices
        N = verts.shape[0] // 100
        grads = []
        verts_split = np.array_split(verts, N)
        for verts_batch in verts_split:
            grad = self.sdf_network.gradient(torch.as_tensor(verts_batch).float().cuda())
            grads.append(grad.detach().cpu().numpy())
        grads = np.concatenate(grads, axis=0)
        mesh.vertex_normals = grads
        mesh = mesh.process(validate=True)
        return mesh
    def extract_fields(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_fields(bound_min,
                                bound_max,
                                resolution=resolution,
                                query_func=lambda pts: self.sdf_network.sdf(pts))
