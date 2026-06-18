from ast import Lambda
from math import prod
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedder import get_embedder
from models.fields import MLP, SDFNetwork
from numpy import pi
import tinycudann as tcnn
import math
import numpy as np
def _format_brdf_params(param, newdim=True):
    '''
    param: tensor of shape (..., n_lobes)
    return: reshaped tensor of shape (n_lobes, ..., 1) if newdim, otherwise (n_lobes, ...)
    '''
    if newdim:
        return param.movedim(-1, 0).unsqueeze(-1)
    else:
        return param.movedim(-1, 0)

def unpack_brdf_params_GGX(brdf_params, n_lobes=None, trichromatic=False, constant_r0=False):
    '''
    unpack brdf_params with given format:
    args: 
        brdf_params : tensor of shape (..., D)
    returns:
        diffuse_albedo: tensor of shape (..., 3)
        specular_albedo: tensor of shape (n_lobes, ..., 3) or (n_lobes, ..., 1), depending on whether the model is trichromatic
        roughness: tensor of shape (n_lobes, ..., 1)
        r0: tensor of shape (n_lobes, ..., 1)
    '''
    if trichromatic:
        n_channels = 3
    else:
        n_channels = 1

    if n_lobes is None:
        n_lobes = (brdf_params-3) // (2+n_channels)
    
    assert (n_lobes*(2+n_channels) + 3 == brdf_params.shape[-1])

    diffuse_albedo, specular_albedo, roughness, r0 = brdf_params.split([3, n_lobes*n_channels, n_lobes, n_lobes], -1)

    diffuse_albedo = F.relu(diffuse_albedo)
    specular_albedo = F.relu(specular_albedo).reshape(specular_albedo.shape[:-1] + (n_channels, n_lobes))
    roughness = F.sigmoid(roughness)
    if constant_r0:
        r0 = F.sigmoid(r0)
    else:
        r0 = torch.ones_like(r0)
    print("r0", r0)
    return diffuse_albedo, _format_brdf_params(specular_albedo, False), _format_brdf_params(roughness), _format_brdf_params(r0)


def unpack_brdf_params_burley(brdf_params, brdf_config):
    '''
    unpack brdf_params with given format:
    args: 
        brdf_params : tensor of shape (..., D)
    returns:
        diffuse_albedo: tensor of shape (..., 3)
        specular_albedo: tensor of shape (n_lobes, ..., 3) or (n_lobes, ..., 1), depending on whether the model is trichromatic
        roughness: tensor of shape (n_lobes, ..., 1)
        r0: tensor of shape (n_lobes, ..., 1)
    '''
    # [0,1]
    # subsurface  1
    # metallic  1
    # specular 1
    # clearcoat 1
    # roughness 1
    # clearcoat_gloss 1
    # base_color 3

    if (not ("no_sigmoid" in brdf_config)) or (not brdf_config["no_sigmoid"]):
        print("applying sigmoid")
        brdf_params = F.sigmoid(brdf_params)

    subsurface, metallic, specular, clearcoar, roughness, clearcoat_gloss, base_color = torch.split(brdf_params, [1,1,1,1,1,1,3], dim=-1)

    return subsurface*brdf_config.get("subsurface", 1.0),\
           metallic*brdf_config.get("metallic", 1.0),\
           specular*brdf_config.get("specular", 1.0),\
           clearcoar*brdf_config.get("clearcoar", 1.0),\
           roughness, clearcoat_gloss, base_color






def dot(tensor1, tensor2, dim=-1, keepdim=False, non_negative=False, epsilon=1e-6) -> torch.Tensor:
    x =  (tensor1 * tensor2).sum(dim=dim, keepdim=keepdim)
    if non_negative:
        x = torch.clamp_min(x, epsilon)
    return x

def _GGX_smith(hz, roughness, epsilon=1e-10):
    hz_sq = hz**2
    roughness_sq = roughness**2
    D = roughness_sq / pi / (hz_sq * (roughness_sq-1) + 1 + epsilon)**2 # GGX
    G = 2 / ( torch.sqrt(1 + roughness_sq * (1/hz_sq - 1)) + 1)

    return D, G

def _CC_smith(hz, roughness):
    hz_sq = hz**2
    roughness_sq = roughness**2
    D = (roughness_sq-1) / (pi*2*torch.log(roughness)*(1+(roughness_sq-1)*hz_sq))
    G = 2 / ( torch.sqrt(1 + 0.0625 * (1/hz_sq - 1)) + 1)

    return D, G

def _diffuse(hz, roughness, subsurface):
    F_D90 = 2*roughness + 0.5
    base_diffuse = (1 + (F_D90 - 1)*((1-hz)**5))**2 / pi

    F_SS = (1 + (roughness - 1)*((1-hz)**5))
    subsurface_diffuse = 1.25 / pi * (F_SS**2 * (0.5/(hz*0.9999+0.0001)-0.5) + 0.5)

    return (1-subsurface)*base_diffuse + subsurface*subsurface_diffuse



def _GGX_shading(normal_vecs, incident_vecs, view_vecs, roughness, r0=None, epsilon=1e-6):
    '''
    normal_vecs, incident_vecs, view_vecs: (...,3) normalised vectors
    roughness: (k_lobes, ..., 1) rms slope
    r0: (k_lobes, ..., 1)fresnel factor
    returns: (...,k_lobes) specular factors
    '''
    half_vecs = torch.nn.functional.normalize(incident_vecs+view_vecs, dim=-1)
    
    roughness = 0.0001 + (roughness) * (1-epsilon-0.0001)
    # Beckmann model for D
    h_n = dot(half_vecs, normal_vecs, non_negative=True, keepdim=True) # (..., 1)
    # cos_alpha_sq = h_n**2 # (...)
    # cos_alpha_sq = cos_alpha_sq.unsqueeze(dim=-1) # (..., 1)
    # cos_alpha_r_sq = torch.clamp_min(cos_alpha_sq*(roughness**2), epsilon) # ([k_lobes,] ..., 1)
    # # D = torch.exp( (cos_alpha_sq - 1) /  cos_alpha_r_sq ) / \
    # #     ( np.pi * cos_alpha_r_sq * cos_alpha_sq ) # ([k_lobes,] ..., 1)  # Beckmann

    # # GGX model
    # roughness_sq = roughness**2
    # D = roughness_sq / pi / (cos_alpha_sq * (roughness_sq-1) + 1 + epsilon)**2 # GGX

    # # Geometric term G
    # v_n = dot(view_vecs, normal_vecs, non_negative=True) # (...)
    v_h = dot(half_vecs, view_vecs, non_negative=True, keepdim=True) # (..., 1)
    # i_n = dot(incident_vecs, normal_vecs, non_negative=True) # (...)

    v_n = h_n
    i_n = h_n

    # # G = torch.clamp_max(torch.min(i_n, v_n) * 2 * h_n / v_h, 1) # (...)
    # # G = G.unsqueeze(dim=-1) # (..., 1)
    
    # # GGX
    # mask_G = (v_h > 0).float().unsqueeze(dim=-1) # (..., 1)
    # G = 2 / ( torch.sqrt(1 + roughness_sq * (1/v_n.unsqueeze(dim=-1)**2 - 1)) + torch.sqrt(1 + roughness_sq * (1/i_n.unsqueeze(dim=-1)**2 - 1)) )
    # G = G * mask_G

    D, G = _GGX_smith(h_n, roughness, epsilon)

    # Schlick's approximation for F
    if r0 is None:
        F = 1
    else:
        F = r0 + (1-r0) * ((1 - v_h) ** 5) # ([k_lobes,] ..., 1)

    ret = (D*F*G) / (pi*i_n*v_n+epsilon) # ([k_lobes,] ..., 1)

    return ret

def _burley_shading(normal_vecs, incident_vecs, view_vecs, brdf_params, brdf_config):

    half_vecs = torch.nn.functional.normalize(incident_vecs+view_vecs, dim=-1)
    h_n = dot(half_vecs, normal_vecs, non_negative=True, keepdim=True) # (..., 1)
    
    subsurface, metallic, specular, clearcoat, roughness, clearcoat_gloss, base_color = unpack_brdf_params_burley(brdf_params, brdf_config)

    clearcoat_roughness = 0.1 - 0.099 * clearcoat_gloss
    alpha = 0.0001 + (roughness**2) * (1-0.0002)

    D_metal, G_metal = _GGX_smith(h_n, alpha) #(..., 1)
    D_clearcoat, G_clearcoat = _CC_smith(h_n, clearcoat_roughness) #(..., 1)

    if brdf_config.get("mask_shadowing", "joint") == "independent":
        G_metal = G_metal * G_metal
        G_clearcoat = G_clearcoat * G_clearcoat
    
    F_metal = (1-metallic)*specular*0.08 + metallic*base_color # (..., 3)
    F_clearcoat = 0.04

    r_specular = D_metal * G_metal * F_metal / (4 * h_n * h_n) # (..., 3)
    r_clearcoat = D_clearcoat * G_clearcoat * F_clearcoat / (4 * h_n * h_n) # (..., 1)
    r_diffuse = _diffuse(h_n, roughness, subsurface) * base_color #(..., 3)

    return (1-metallic)*r_diffuse, (r_specular + 0.25*clearcoat*r_clearcoat)


def _apply_lighting_GGX(points, normals, view_dirs, light_dirs, irradiance, diffuse_albedo, specular_albedo, roughness, r0):
    """
    Args:
        points: torch tensor of shape (..., 3).
        normals: torch tensor of shape (..., 3), from inside to outside, only directions matter
        view_dirs: torch tensor of shape (..., 3), from viewpoint to object, only directions matter
        light_dirs: torch tensor of shape (..., 3), from light to object, only directions matter
        irradiance: torch tensor of shape (..., 3) or (..., 1), nonnegative
        diffuse_albedo: torch tensor of shape (..., 3) or (..., 1)
        diffuse_albedo: torch tensor of shape ([k_lobes], ..., 3) or ([k_lobes], ..., 1)
        roughness: torch tensor of shape ([k_lobes], ..., 1)
        r0: torch tensor of shape ([k_lobes], ..., 1)

    Returns:
        diffuse_color: # (..., 3) or (..., 1)
        specular_color: # ([k_lobes,] ..., 3) or ([k_lobes,] ..., 1)
    """
    normals = F.normalize(normals, dim=-1)
    light_dirs_ = F.normalize(light_dirs, dim=-1)
    view_dirs = F.normalize(view_dirs, dim=-1)


    falloff = F.relu(-(normals * light_dirs_).sum(-1)) # (...)
    forward_facing = dot(normals, view_dirs) < 0
    visible_mask = ((falloff > 0) & forward_facing) # (...) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (...) cosine falloff, 0 if not visible

    specular_reflectance = _GGX_shading(normals, -light_dirs_, -view_dirs, roughness, r0) * specular_albedo # ([k_lobes,] ..., 3) or ([k_lobes,] ..., 1)
    irradiance = torch.unsqueeze(falloff, dim=-1) * irradiance  # (..., 3) or (..., 1)
    
    diffuse_color = diffuse_albedo * irradiance # (..., 3) or (..., 1)
    specular_color = specular_reflectance * irradiance # ([k_lobes,] ..., 3) or ([k_lobes,] ..., 1)

    return diffuse_color, specular_color


def _apply_shading_burley(points, normals, view_dirs, light_dirs, irradiance, brdf_params, brdf_config):
    normals = F.normalize(normals, dim=-1)
    light_dirs_ = F.normalize(light_dirs, dim=-1)
    view_dirs = F.normalize(view_dirs, dim=-1)


    falloff = F.relu(-(normals * light_dirs_).sum(-1)) # (...)
    forward_facing = dot(normals, view_dirs) < 0
    visible_mask = ((falloff > 0) & forward_facing) # (...) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (...) cosine falloff, 0 if not visible
    irradiance = torch.unsqueeze(falloff, dim=-1) * irradiance  # (..., 3) or (..., 1)

    diffuse, non_diffuse = _burley_shading(normals, -light_dirs_, -view_dirs, brdf_params, brdf_config)
    # print("burley diffuse", diffuse)
    # print("irradiance", irradiance)
    return diffuse*irradiance, non_diffuse*irradiance

def _apply_diffuse(points, normals, view_dirs, light_dirs, irradiance, diffuse_albedo):
    normals = F.normalize(normals, dim=-1)
    light_dirs_ = F.normalize(light_dirs, dim=-1)
    view_dirs = F.normalize(view_dirs, dim=-1)


    falloff = F.relu(-(normals * light_dirs_).sum(-1)) # (...)
    forward_facing = dot(normals, view_dirs) < 0
    visible_mask = ((falloff > 0) & forward_facing) # (...) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (...) cosine falloff, 0 if not visible
    irradiance = torch.unsqueeze(falloff, dim=-1) * irradiance  # (..., 3) or (..., 1)
    # print("falloff, irradiance", falloff, irradiance)
    # exit()
    diffuse_color = diffuse_albedo * irradiance # (..., 3) or (..., 1)
    return diffuse_color

def _apply_dist_only(points, normals, view_dirs, light_dirs, irradiance, diffuse_albedo):
    normals = F.normalize(normals, dim=-1)
    light_dirs_ = F.normalize(light_dirs, dim=-1)
    view_dirs = F.normalize(view_dirs, dim=-1)


    falloff = F.relu(-(normals * light_dirs_).sum(-1)) # (...)
    forward_facing = dot(normals, view_dirs) < 0
    visible_mask = ((falloff > 0) & forward_facing) # (...) boolean
    falloff = torch.where(visible_mask, falloff, torch.zeros(1, device=falloff.device)) # (...) cosine falloff, 0 if not visible
    irradiance = torch.unsqueeze(falloff, dim=-1) * irradiance  # (..., 3) or (..., 1)
    # print("falloff, irradiance", falloff, irradiance)
    # exit()
    diffuse_color = diffuse_albedo * irradiance # (..., 3) or (..., 1)
    return diffuse_color
def safe_exp(x):
    return torch.exp(torch.clamp(x, max=10.0))

class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
            # self.embedview_fn = tcnn.Encoding(
            #     n_input_dims=3,
            #     encoding_config={
            #         "otype": "SphericalHarmonics",
            #         "degree": multires_view,
            #     },
            # )
            # dims[0] += (self.embedview_fn.n_output_dims - 3)
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
    def set_encoding_level(self, perc_iters):
        pass
    def forward(self, points, normals, view_dirs, brdf_params, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
            
        rendering_input = None
        # This rendering network does not consume BRDF parameters.
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                
        # print("rendering net prediction pre squeeze", x.min(), x.max())
        if self.squeeze_out:
            #x = torch.relu(x)
            # x = torch.sigmoid(x)
            
            # x = F.softplus(x)
            x = safe_exp(x)
        # print("rendering net prediction", x.min(), x.max())
        return x
class EnvironmentNetwork(nn.Module):
    def __init__(self,
                 d_in=3,
                 d_out=3,
                 d_hidden=256,
                 n_layers=2,
                 weight_norm=True,
                 multires_view=4,
                 squeeze_out=True):
        super().__init__()

        self.squeeze_out = squeeze_out
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
            # self.embedview_fn = tcnn.Encoding(
            #     n_input_dims=3,
            #     encoding_config={
            #         "otype": "SphericalHarmonics",
            #         "degree": multires_view,
            #     },
            # )
            # dims[0] += (self.embedview_fn.n_output_dims - 3)
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
    def set_encoding_level(self, perc_iters):
        pass
    def forward(self, view_dirs):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
            
        rendering_input = None
        # This rendering network does not consume BRDF parameters.
        rendering_input = torch.cat([view_dirs], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                
        # print("rendering net prediction pre squeeze", x.min(), x.max())
        if self.squeeze_out:
            #x = torch.relu(x)
            # x = torch.sigmoid(x)
            
            # x = F.softplus(x)
            x = safe_exp(x)
        # print("rendering net prediction", x.min(), x.max())
        return x
class RenderingNetworkLight(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires=0,
                 multires_view=0,
                 multires_light=0,
                 light_features=3,
                 squeeze_out=True,
                 skip_in=()):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.embed_fn = None
        self.embedview_fn = None
        self.embedlight_fn = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)
        
        if multires_view > 0:
            # embedview_fn, input_ch = get_embedder(multires_view)
            # self.embedview_fn = embedview_fn
            # dims[0] += (input_ch - 3)
            self.embedview_fn = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": multires_view,
                },
            )
            dims[0] += (self.embedview_fn.n_output_dims - 3)
        if multires_light > 0:
            embedlight_fn, input_ch = get_embedder(multires_light, input_dims=light_features)
            self.embedlight_fn = embedlight_fn
            dims[0] += (input_ch - light_features)
            
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        
        
    def set_encoding_level(self, perc_iters):
        pass
    def forward(self, points, normals, dirs, light_pos, feature_vectors):
        if self.embed_fn is not None:
            points = self.embed_fn(points)
            pass
        
        refdirs = 2.0 * torch.sum(normals * -dirs, axis=-1, keepdims=True) * normals + dirs
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(dirs)
            refdirs = self.embedview_fn(refdirs)
            
        if self.embedlight_fn is not None:
            light_pos = self.embedlight_fn(light_pos)
            
        rendering_input = None
        # assert self.mode == "idr"
        # print("pts", points, view_dirs, normals, light_pos, feature_vectors)
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, light_pos, feature_vectors], dim=-1)
        # elif self.mode == 'no_view_dir':
        #     rendering_input = torch.cat([points, normals, brdf_params, feature_vectors], dim=-1)
        # elif self.mode == 'no_normal':
        #     rendering_input = torch.cat([points, view_dirs, brdf_params, feature_vectors], dim=-1)
        elif self.mode == "idr_refdir":
            rendering_input = torch.cat([points, refdirs, light_pos, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir_no_normal':
            rendering_input = torch.cat([points, feature_vectors])
        else:
            raise NotImplementedError(self.mode)
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                
        # print("rendering net prediction pre squeeze", x.min(), x.max())
        if self.squeeze_out:
            #x = torch.relu(x)
            # x = torch.sigmoid(x)
            # x = F.softplus(x)
            x = safe_exp(x)
            # x = 1 / (F.softplus(x) + 1e-6)
        # print("rendering net prediction", x.min(), x.max())
        return x
class RenderingNetworkLightNGP(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 light_features=1,
                 squeeze_out=True,
                 skip_in=()):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.embed_fn = None
        self.embedview_fn = None
        self.embedlight_fn = None

        self.embedview_fn = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                )
        
        self.embedlight_fn = tcnn.Encoding(
            n_input_dims=1,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
            
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        
        
    def set_encoding_level(self, perc_iters):
        pass
    def forward(self, points, normals, dirs, light_pos, feature_vectors):
        if self.embed_fn is not None:
            points = self.embed_fn(points)
            pass
        
        refdirs = 2.0 * torch.sum(normals * -dirs, axis=-1, keepdims=True) * normals + dirs
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(dirs)
            refdirs = self.embedview_fn(refdirs)
            
        if self.embedlight_fn is not None:
            light_pos = self.embedlight_fn(light_pos)
            
        rendering_input = None
        # assert self.mode == "idr"
        # print("pts", points, view_dirs, normals, light_pos, feature_vectors)
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, light_pos, feature_vectors], dim=-1)
        # elif self.mode == 'no_view_dir':
        #     rendering_input = torch.cat([points, normals, brdf_params, feature_vectors], dim=-1)
        # elif self.mode == 'no_normal':
        #     rendering_input = torch.cat([points, view_dirs, brdf_params, feature_vectors], dim=-1)
        elif self.mode == "idr_refdir":
            rendering_input = torch.cat([points, refdirs, light_pos, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir_no_normal':
            rendering_input = torch.cat([points, feature_vectors])
        else:
            raise NotImplementedError(self.mode)
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                
        # print("rendering net prediction pre squeeze", x.min(), x.max())
        if self.squeeze_out:
            #x = torch.relu(x)
            # x = torch.sigmoid(x)
            # x = F.softplus(x)
            x = safe_exp(x)
            # x = 1 / (F.softplus(x) + 1e-6)
        # print("rendering net prediction", x.min(), x.max())
        return x
class RenderingNetworkLightPlus(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires=0,
                 multires_view=0,
                 multires_light=0,
                 light_features=3,
                 squeeze_out=True,
                 skip_in=()):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        assert self.squeeze_out
        self.d_out = d_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out*2]
        
        self.embed_fn = None
        self.embedview_fn = None
        self.embedlight_fn = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)
        
        if multires_view > 0:
            # embedview_fn, input_ch = get_embedder(multires_view)
            # self.embedview_fn = embedview_fn
            # dims[0] += (input_ch - 3)
            self.embedview_fn = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": multires_view,
                },
            )
            print(self.embedview_fn.n_output_dims)
            dims[0] += (self.embedview_fn.n_output_dims - 3)
        if multires_light > 0:
            embedlight_fn, input_ch = get_embedder(multires_light, input_dims=light_features)
            self.embedlight_fn = embedlight_fn
            dims[0] += (input_ch - light_features)
            
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
    def set_encoding_level(self, perc_iters):
        pass
    def forward(self, points, normals, dirs, light_pos, feature_vectors):
        if self.embed_fn is not None:
            points = self.embed_fn(points)
        refdirs = 2.0 * torch.sum(normals * -dirs, axis=-1, keepdims=True) * normals + dirs
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(dirs)
            refdirs = self.embedview_fn(refdirs)
            

        if self.embedlight_fn is not None:
            light_pos = self.embedlight_fn(light_pos)
            
        rendering_input = None
        # assert self.mode == "idr"
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, light_pos, feature_vectors], dim=-1)
        # elif self.mode == 'no_view_dir':
        #     rendering_input = torch.cat([points, normals, brdf_params, feature_vectors], dim=-1)
        # elif self.mode == 'no_normal':
        #     rendering_input = torch.cat([points, view_dirs, brdf_params, feature_vectors], dim=-1)
        elif self.mode == "idr_refdir":
            rendering_input = torch.cat([points, refdirs, light_pos, feature_vectors], dim=-1)
        else:
            raise NotImplementedError(self.mode)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                pass
            pass
        # x = F.softplus(x)
        # x = F.sigmoid(x) #N, 6
        # x = F.sigmoid(x)
        # x = F.
        # print(x.shape)
        x = safe_exp(x)
        x1 = x[:, :self.d_out]
        x2 = x[:, self.d_out:]
        x = x1/(x2+1e-6)
        # x = safe_exp(x)
        
        # print("x1", x1.min(), x1.max(), x1.mean())
        # print("x2", x2.min(), x2.max(), x2.mean())
        # print("x", x.min(), x.max(), x.mean())
        return x

class RenderingNetworkNGP(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        # if multires_view > 0:
        #     embedview_fn, input_ch = get_embedder(multires_view)
        #     self.embedview_fn = embedview_fn
        #     dims[0] += (input_ch - 3)
        self.embedview_fn = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.encoding_length = 16
        self.base_enabled_encoding = 4
        self.enabled_encoding = self.encoding_length
        
        dims[0] += self.encoding_length - 3
        self.num_layers = len(dims)

        # for l in range(0, self.num_layers - 1):
        #     out_dim = dims[l + 1]
        #     lin = nn.Linear(dims[l], out_dim)

        #     if weight_norm:
        #         lin = nn.utils.weight_norm(lin)

        #     setattr(self, "lin" + str(l), lin)
        self.mlp = tcnn.Network(
            n_input_dims=dims[0],
            n_output_dims=d_out,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": d_hidden,
                "n_hidden_layers": n_layers,
            },
        )
        # self.mlp = torch.jit.trace(MLP(
        #     d_in=dims[0],
        #     d_out=d_out,
        #     d_hidden=d_hidden,
        #     n_layers=n_layers,
        #     scale=None,
        #     bias=None,
        #     geometric_init=False,
        #     inside_outside=False
        # ), torch.rand(16, dims[0]))

        # assert not self.squeeze_out
        
        
        self.relu = nn.ReLU()
    def set_encoding_level(self, perc_iters):
        # if perc_iters > 0.1:
        #     perc_iters -= 0.1
        #     perc_iters = perc_iters / 0.9
        #     self.enabled_encoding = min(self.base_enabled_encoding + math.floor((self.encoding_length) * perc_iters), self.encoding_length)
        # self.enabled_encoding = min(self.base_enabled_encoding + math.floor((self.encoding_length) * perc_iters), self.encoding_length)

        # fac = 1
        # new_encoding = min(self.base_enabled_encoding + math.floor((self.encoding_length) * perc_iters * fac), self.encoding_length)
        # if new_encoding != self.enabled_encoding:
        #     print("rendering new encoding: ", new_encoding)
        # self.enabled_encoding = new_encoding
        self.enabled_encoding = self.encoding_length
        pass
    
    def get_encoding(self, inputs):
        embed = self.embedview_fn(inputs).clone()
        embed[:, self.enabled_encoding:] = 0.0
        return embed
        
    def forward(self, points, normals, view_dirs, brdf_params, feature_vectors):
        # if self.embedview_fn is not None:
        #     view_dirs = self.embedview_fn(view_dirs)
        view_dirs = self.get_encoding(view_dirs)    
        rendering_input = None
        
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, brdf_params, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, brdf_params, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, brdf_params, feature_vectors], dim=-1)

        x = rendering_input

        # for l in range(0, self.num_layers - 1):
        #     lin = getattr(self, "lin" + str(l))

        #     x = lin(x)

        #     if l < self.num_layers - 2:
        #         x = self.relu(x)
        # input('before rendering mlp')
        x = self.mlp(x)
        # input('after rendering mlp')
        # print("rendering dtype", x.dtype)
        if self.squeeze_out:
            #x = torch.relu(x)
            x = torch.sigmoid(x)
        return x
class RenderingNetworkIndependent(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        
        self.relu = nn.ReLU()
        self.sdf = SDFNetwork(d_in=3, d_out=1+d_feature, d_hidden=256, n_layers=8, skip_in=(4,),multires=6)
        self.encoding_length = input_ch
        self.base_enabled_encoding = 2
        self.enabled_encoding = self.encoding_length
    def set_encoding_level(self, perc_iters):
        fac = 1
        new_encoding = min(self.base_enabled_encoding + math.floor((self.encoding_length) * perc_iters * fac), self.encoding_length)
        if new_encoding != self.enabled_encoding:
            print("rendering new encoding: ", new_encoding)
        self.enabled_encoding = new_encoding
        pass

    def get_encoding(self, inputs):
        embed = self.embedview_fn(inputs).clone()
        embed[:, self.enabled_encoding:] = 0.0
        return embed

    def forward(self, points, normals, view_dirs, brdf_params, feature_vectors):
        feature_vectors = self.sdf(points)[:, 1:]
        # if self.embedview_fn is not None:
        #     view_dirs = self.embedview_fn(view_dirs)
        view_dirs = self.get_encoding(view_dirs)
        rendering_input = None
        
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, brdf_params, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, brdf_params, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, brdf_params, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            #x = torch.relu(x)
            x = torch.sigmoid(x)
        return x

class RenderingNetworkIRON(nn.Module):
    def __init__(
        self,
        d_feature,
        mode,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        weight_norm=True,
        multires=0,
        multires_view=0,
        squeeze_out=True,
        squeeze_out_scale=1.0,
        output_bias=0.0,
        output_scale=1.0,
        skip_in=(),
    ):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        self.output_bias = output_bias
        self.output_scale = output_scale
        self.squeeze_out_scale = squeeze_out_scale
    def set_encoding_level(self, perc_iters):
        pass
    def forward(self, points, normals, view_dirs, brdf_params, feature_vectors):
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
            
        rendering_input = None
        
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, brdf_params, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, brdf_params, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, brdf_params, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            #x = torch.relu(x)
            x = torch.sigmoid(x)
        return x

    # def forward(self, points, normals, view_dirs, feature_vectors):

    #     if self.embed_fn is not None:
    #         points = self.embed_fn(points)

    #     if self.embedview_fn is not None and self.mode != "no_view_dir":
    #         view_dirs = self.embedview_fn(view_dirs)

    #     rendering_input = None

    #     if self.mode == "idr":
    #         rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
    #     elif self.mode == "no_view_dir":
    #         rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
    #     elif self.mode == "no_normal":
    #         rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

    #     x = rendering_input

    #     for l in range(0, self.num_layers - 1):
    #         lin = getattr(self, "lin" + str(l))

    #         if l in self.skip_in:
    #             x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

    #         x = lin(x)

    #         if l < self.num_layers - 2:
    #             x = self.relu(x)

    #     x = self.output_scale * (x + self.output_bias)
    #     if self.squeeze_out:
    #         x = self.squeeze_out_scale * torch.sigmoid(x)

    #     return x
class RadianceCache(nn.Module):
    def __init__(self,
                 d_out,
                 d_hidden,
                 n_layers,
                 point_encoding,
                 normal_encoding,
                 view_encoding,
                 light_encoding,
                 squeeze_out=True):
        super().__init__()
        self.squeeze_out = squeeze_out

        self.embedview_fn = None
        # if multires_view > 0:
        #     embedview_fn, input_ch = get_embedder(multires_view)
        #     self.embedview_fn = embedview_fn
        #     dims[0] += (input_ch - 3)
        self.embed_fn_point = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": point_encoding["n_levels"],
                "n_features_per_level": point_encoding["n_features_per_level"],
                "log2_hashmap_size": point_encoding["log2_hashmap_size"],
                "base_resolution": point_encoding["base_resolution"],
                "per_level_scale": point_encoding["per_level_scale"],
            },
            # dtype=torch.float32
        )
        self.embed_fn_normal = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": normal_encoding["n_levels"],
                "n_features_per_level": normal_encoding["n_features_per_level"],
                "log2_hashmap_size": normal_encoding["log2_hashmap_size"],
                "base_resolution": normal_encoding["base_resolution"],
                "per_level_scale": normal_encoding["per_level_scale"],
            },
            # dtype=torch.float32
        )
        self.embed_fn_view = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": view_encoding["n_levels"],
                "n_features_per_level": view_encoding["n_features_per_level"],
                "log2_hashmap_size": view_encoding["log2_hashmap_size"],
                "base_resolution": view_encoding["base_resolution"],
                "per_level_scale": view_encoding["per_level_scale"],
            },
            # dtype=torch.float32
        )
        self.embed_fn_light = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": light_encoding["n_levels"],
                "n_features_per_level": light_encoding["n_features_per_level"],
                "log2_hashmap_size": light_encoding["log2_hashmap_size"],
                "base_resolution": light_encoding["base_resolution"],
                "per_level_scale": light_encoding["per_level_scale"],
            },
            # dtype=torch.float32
        )
        dims_0 = self.embed_fn_point.n_output_dims + self.embed_fn_normal.n_output_dims + self.embed_fn_view.n_output_dims + self.embed_fn_light.n_output_dims
        # dims_0 = self.embed_fn_point_ref.n_output_dims + self.embed_fn_light.n_output_dims
        self.mlp = tcnn.Network(
            n_input_dims=dims_0,
            n_output_dims=d_out,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": d_hidden,
                "n_hidden_layers": n_layers,
            },
        )
        
        
        
    def forward(self, points, normals, view_dirs, light_origin, light_lum, brdf_params, feature_vectors):
        view_dirs = F.normalize(view_dirs, dim=-1) 
        normals = F.normalize(normals, dim=-1) 
        # points = self.embed_fn_point_ref(point_ref) # https://github.com/NVlabs/tiny-cuda-nn/issues/286
        points = self.embed_fn_point((points+1)/2)
        normals = self.embed_fn_normal((normals+1)/2)
        view_dirs = self.embed_fn_view((view_dirs+1)/2)
        light_origin = light_origin.unsqueeze(dim=0).repeat(points.shape[0], 1)
        light_origin = self.embed_fn_light((light_origin+1)/2)
        
        rendering_input = torch.cat([points, normals, view_dirs, light_origin], dim=-1)

        x = rendering_input

        x = self.mlp(x)
        if self.squeeze_out:
            x = torch.exp(x)
        x  = light_lum * x
        # print("x", torch.isfinite(x).all())
        return x, {}

class RadianceCacheOcclusion(nn.Module):
    def __init__(self,
                 d_out,
                 d_hidden,
                 n_layers,
                 point_encoding,
                 num_lights=1000,
                 light_features=512,
                 skip_in=()):
        super().__init__()
        weight_norm = True
        self.light_features = light_features
        dims = [light_features] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.embed_fn_point = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": point_encoding["n_levels"],
                "n_features_per_level": point_encoding["n_features_per_level"],
                "log2_hashmap_size": point_encoding["log2_hashmap_size"],
                "base_resolution": point_encoding["base_resolution"],
                "per_level_scale": point_encoding["per_level_scale"],
            },
            # dtype=torch.float32
        )
        dims[0] += self.embed_fn_point.n_output_dims

        self.embedlight_fn = torch.nn.Embedding(num_lights, light_features)

        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        
    def set_encoding_level(self, perc_iters):
        pass

    def forward(self, points, light_idx):
        points = self.embed_fn_point(points)

        light_features = self.embedlight_fn(light_idx)
        # print("light feature size", light_features.shape)
        rendering_input = torch.cat([points, light_features], dim=-1)
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                
        # print("rendering net prediction", x.min(), x.max())
        return x


class RadianceCacheMLPOcclusion(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires=0,
                 multires_view=0,
                 multires_normal=0,
                 multires_light=0,
                 light_features=4,
                 squeeze_out=True,
                 skip_in=()):
        super().__init__()
        self.light_features = light_features
        if self.light_features == 0:
            assert multires_light == 0

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.embed_fn = None
        self.embedview_fn = None
        self.embednormal_fn = None
        self.embedlight_fn = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)
        if multires_view > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embedview_fn = embed_fn
            dims[0] += (input_ch - 3)
        if multires_normal > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embednormal_fn = embed_fn
            dims[0] += (input_ch - 3)

        # if multires_light > 0:
        #     embedlight_fn, input_ch = get_embedder(multires_light, input_dims=light_features)
        #     self.embedlight_fn = embedlight_fn
        #     dims[0] += (input_ch - light_features)
        self.embedlight_fn = torch.nn.Embedding(multires_light, 512)
        # dims[0] += (512 - light_features)
        dims[0] += 512

        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        
        
    def set_encoding_level(self, perc_iters):
        pass
    def forward(self, points, normals, view_dirs, light_origin, light_lum, brdf_params, feature_vectors):
        # light_dir = points - light_origin
        if self.embed_fn is not None:
            points = self.embed_fn(points)
            pass
        
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        if self.embednormal_fn is not None:
            normals = self.embednormal_fn(normals)  


        # if self.light_features > 0:
        assert self.light_features == 4
        # light_dist = (light_dir*light_dir).sum(-1,keepdim=True)
        # light_dir = light_dir / (light_dist + 1e-8)
        # light_features = torch.cat([light_dir,
        #                            light_dist], dim=-1)
        # if self.embedlight_fn is not None:
        #     light_features = self.embedlight_fn(light_features)
        light_features = self.embedlight_fn(light_origin)
        # print("light feature size", light_features.shape)
        rendering_input = None
        assert self.mode == "idr"
        # print(points.shape, view_dirs.shape, normals.shape, light_features.shape)
        if self.mode == 'idr':
            print("light_features", self.light_features, self.light_features > 0)
            # if self.light_features > 0:
            # print("=============light feature included")
            rendering_input = torch.cat([points, view_dirs, normals, light_features], dim=-1)
            # else:
                # rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
        # elif self.mode == 'no_view_dir':
        #     rendering_input = torch.cat([points, normals, brdf_params, feature_vectors], dim=-1)
        # elif self.mode == 'no_normal':
        #     rendering_input = torch.cat([points, view_dirs, brdf_params, feature_vectors], dim=-1)
        elif self.mode == "idr_refdir":
            rendering_input = torch.cat([points, refdirs, light_features], dim=-1)
        elif self.mode == 'no_view_dir_no_normal':
            rendering_input = torch.cat([points])
        else:
            raise NotImplementedError(self.mode)
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                
        assert not self.squeeze_out
        # print("rendering net prediction", x.min(), x.max())
        return x, {}
class RadianceCacheMLP(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires=0,
                 multires_view=0,
                 multires_normal=0,
                 multires_light=0,
                 light_features=3,
                 squeeze_out=True,
                 skip_in=()):
        super().__init__()
        self.light_features = light_features
        if self.light_features == 0:
            assert multires_light == 0

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.embed_fn = None
        self.embedview_fn = None
        self.embednormal_fn = None
        self.embedlight_fn = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)
        print("ch1", input_ch)
        if multires_view > 0:
            # embedview_fn, input_ch = get_embedder(multires_view)
            # self.embedview_fn = embedview_fn
            # dims[0] += (input_ch - 3)
            embed_fn, input_ch = get_embedder(multires)
            self.embedview_fn = embed_fn
            dims[0] += (input_ch - 3)
            # self.embedview_fn = tcnn.Encoding(
            #     n_input_dims=3,
            #     encoding_config={
            #         "otype": "SphericalHarmonics",
            #         "degree": multires_view,
            #     },
            # )
            # dims[0] += (self.embedview_fn.n_output_dims - 3)
        # print("ch2", self.embedview_fn.n_output_dims)
        if multires_normal > 0:
            # embedview_fn, input_ch = get_embedder(multires_view)
            # self.embedview_fn = embedview_fn
            # dims[0] += (input_ch - 3)
            embed_fn, input_ch = get_embedder(multires)
            self.embednormal_fn = embed_fn
            dims[0] += (input_ch - 3)
            # self.embednormal_fn = tcnn.Encoding(
            #     n_input_dims=3,
            #     encoding_config={
            #         "otype": "SphericalHarmonics",
            #         "degree": multires_normal,
            #     },
            # )
            # dims[0] += (self.embednormal_fn.n_output_dims - 3)
        # print("ch3", self.embednormal_fn.n_output_dims)
        if multires_light > 0:
            embedlight_fn, input_ch = get_embedder(multires_light, input_dims=light_features)
            self.embedlight_fn = embedlight_fn
            dims[0] += (input_ch - light_features)
        print("ch4", input_ch)
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        
        
    def set_encoding_level(self, perc_iters):
        pass
    def forward(self, points, normals, view_dirs, light_origin, light_lum, brdf_params, feature_vectors):
        light_dir = points - light_origin
        if self.embed_fn is not None:
            points = self.embed_fn(points)
            pass
        
        refdirs = 2.0 * torch.sum(normals * -view_dirs, axis=-1, keepdims=True) * normals + view_dirs
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
            refdirs = self.embedview_fn(refdirs)
        if self.embednormal_fn is not None:
            normals = self.embednormal_fn(normals)  


        # light_pos = light_pos.unsqueeze(dim=0).repeat(points.shape[0], 1)
        if self.light_features > 0:
            light_features = 1/(light_dir*light_dir).sum(-1,keepdim=True)
            if self.embedlight_fn is not None:
                light_features = self.embedlight_fn(light_features)
            
        rendering_input = None
        assert self.mode == "idr"
        # print(points.shape, view_dirs.shape, normals.shape, light_features.shape)
        if self.mode == 'idr':
            if self.light_features > 0:
                rendering_input = torch.cat([points, view_dirs, normals, light_features], dim=-1)
            else:
                rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
        # elif self.mode == 'no_view_dir':
        #     rendering_input = torch.cat([points, normals, brdf_params, feature_vectors], dim=-1)
        # elif self.mode == 'no_normal':
        #     rendering_input = torch.cat([points, view_dirs, brdf_params, feature_vectors], dim=-1)
        elif self.mode == "idr_refdir":
            rendering_input = torch.cat([points, refdirs, light_features], dim=-1)
        elif self.mode == 'no_view_dir_no_normal':
            rendering_input = torch.cat([points])
        else:
            raise NotImplementedError(self.mode)
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                
        # print("rendering net prediction pre squeeze", x.min(), x.max())
        if self.squeeze_out:
            #x = torch.relu(x)
            # x = torch.sigmoid(x)
            # x = F.softplus(x)
            x = safe_exp(x)
            # x = 1 / (F.softplus(x) + 1e-6)
        # print("rendering net prediction", x.min(), x.max())
        return x, {}
class PhysicalRenderingNetwork(nn.Module):
    def __init__(self,
                 config,
                 brdf_config,
                 view_net_type="default"
                 ):
        super().__init__()

        self.n_lobes = brdf_config.get("n_specular_lobes", None)
        self.trichromatic = brdf_config.get("type") != "GGX" or brdf_config.get("trichromatic_specular", False)
        self.constant_r0 = brdf_config.get("ignore_Fresnel", False)
        self.no_nvs_grad = config.get("no_grad", False)
        if view_net_type=="default":
            print("default rendering net")
            self.ambient_net = RenderingNetwork(**config["ambient_network"])
        elif view_net_type == "ngp":
            print("ngp_rendering net")
            self.ambient_net = RenderingNetworkNGP(**config["ambient_network"])
        elif view_net_type == "independent":
            print("indepdenet rendering net")
            self.ambient_net = RenderingNetworkIndependent(**config["ambient_network"])
        elif view_net_type == "iron":
            print("iron net")
            self.ambient_net = RenderingNetworkIRON(**config["ambient_network"])

        else:
            raise NotImplementedError()
        
        self.bsdf_type = brdf_config['type']
        if self.bsdf_type == "ambient_sep" or self.bsdf_type == "ambient_sep_direct" or self.bsdf_type == "ambient_sep_wildlight":
            self.light_net =  RenderingNetworkLight(**config["light_network"])
            print(self.light_net)
            pass
        elif self.bsdf_type == "ambient_sep_plus" or self.bsdf_type == "ambient_sep_plus_wildlight":
            self.light_net =  RenderingNetworkLightPlus(**config["light_network"])
            pass
        
        self.bsdf_config = brdf_config
        self.is_darkroom = config.get("darkroom", False)
        print("self.is_darkroom", self.is_darkroom)
        self.n_brdf_dim = brdf_config.dims

        self.gamma = nn.Parameter(config["gamma"]*torch.ones(1, dtype=torch.float32), requires_grad=True)
        #was 0.01
    def flash_light_gamma(self):
        g = self.gamma
        m = torch.clamp(g, min=-1.0, max=1.0)
        return m * (g - 0.5*m)

    def forward(self, points, normals, view_dirs, light_origin, light_lum, brdf_params, feature_vectors):
        
        # print("WARNING normal is detached in physical shader")
        # normals = normals.detach()
        light_dir = points - light_origin
        irradiance = light_lum / (light_dir*light_dir).sum(-1,keepdim=True)
        # print("light_lum", light_lum, "fall off", (light_dir*light_dir).sum(-1,keepdim=True))
        if self.bsdf_type == "GGX":
            diffuse_albedo, specular_albedo, roughness, r0 = unpack_brdf_params_GGX(brdf_params, self.n_lobes, self.trichromatic, self.constant_r0)
            diffuse_active_color, specular_active_color = _apply_lighting_GGX(
                                            points, normals, view_dirs, light_dir, irradiance,
                                            diffuse_albedo, specular_albedo, roughness, r0
                                            )
            specular_active_color = specular_active_color.sum(0)

        elif self.bsdf_type == "Burley":
            diffuse_active_color, specular_active_color = \
                _apply_shading_burley(points, normals, view_dirs, light_dir, irradiance, brdf_params, self.bsdf_config)
        elif self.bsdf_type == "diffuse":
            diffuse_active_color = _apply_diffuse(points, normals, view_dirs, light_dir, irradiance, brdf_params)
            specular_active_color = 0
            pass
        elif self.bsdf_type == "ambient":
            ambient_net_color = self.ambient_net(
                                            points, 
                                            normals, 
                                            view_dirs, 
                                            brdf_params, 
                                            feature_vectors
                                        )
            # print(ambient_net_color.min(), ambient_net_color.max())
            # diffuse_active_color = irradiance * ambient_net_color
            diffuse_active_color = _apply_diffuse(points, normals, view_dirs, light_dir, irradiance, ambient_net_color)
            # diffuse_active_color = ambient_net_color
            specular_active_color = 0
        elif self.bsdf_type == "ambient_sep" or self.bsdf_type == "ambient_sep_wildlight":
            # ambient_net_color = self.ambient_net(
            #                                 points, 
            #                                 normals, 
            #                                 view_dirs, 
            #                                 brdf_params, 
            #                                 feature_vectors
            #                             )
            # # print("light_orig", light_origin.shape, light_origin.unsqueeze(dim=0).repeat(points.shape[0], 1).shape)
            # light_features = torch.cat([light_origin.unsqueeze(dim=0).repeat(points.shape[0], 1), 1/(light_dir*light_dir).sum(-1,keepdim=True)], dim=1)
            # print(light_features.shape)
            # light_features = light_origin.unsqueeze(dim=0).repeat(points.shape[0], 1)
            light_features = 1/(light_dir*light_dir).sum(-1,keepdim=True)
            # light_features = (light_dir*light_dir).sum(-1,keepdim=True)
            # # light_features = (light_dir*light_dir).sum(-1,keepdim=True)
            # print("light features", light_features)
            light_net_color = self.light_net(
                points, 
                normals, 
                view_dirs, 
                light_features, # 3 + 3 + 1 = 7 
                feature_vectors
            )
            # print("light_net_color", light_net_color)
            # print(light_net_color.shape)
            # colocated, gi = torch.split(light_net_color, 3, dim=-1)
            # diffuse_active_color = _apply_diffuse(points, normals, view_dirs, light_dir, irradiance, colocated) + gi
            # print("numerator", colocated.min(), colocated.max(), colocated.mean())
            # print("denominator", gi.min(), gi.max(), gi.mean())
            # diffuse_active_color = colocated / gi
            
            # diffuse_active_color = 1/gi
            # diffuse_active_color = _apply_diffuse(points, normals, view_dirs, light_dir, irradiance, ambient_net_color) + light_net_color
            # diffuse_active_color = _apply_diffuse(points, normals, view_dirs, light_dir, irradiance, ambient_net_color)
            diffuse_active_color = light_net_color
            if self.bsdf_type == "ambient_sep_wildlight":
                print("ambient_sep_wildlight light energy", light_lum)
                diffuse_active_color = diffuse_active_color * light_lum
            # print(ambient_net_color.min(), ambient_net_color.max(), ambient_net_color.mean(), light_net_color.min(), light_net_color.max(), light_net_color.mean(), self.flash_light_gamma())
            # print(light_net_color.min(), light_net_color.max(), light_net_color.mean(), self.flash_light_gamma())
            # diffuse_active_color = ambient_net_color
            specular_active_color = 0
            # diffuse_active_color, specular_active_color = \
            #     _apply_shading_burley(points, normals, view_dirs, light_dir, irradiance, brdf_params, self.bsdf_config)
        elif self.bsdf_type=="ambient_sep_plus":
            # print("this branch")
            light_features = (light_dir*light_dir).sum(-1,keepdim=True)
            light_net_color = self.light_net(
                points, 
                normals, 
                view_dirs, 
                light_features, # 3 + 3 + 1 = 7 
                feature_vectors
            )
            diffuse_active_color = light_net_color
            specular_active_color = 0
        elif self.bsdf_type=="ambient_sep_plus_wildlight":
            light_features = (light_dir*light_dir).sum(-1,keepdim=True)
            light_net_color = self.light_net(
                points, 
                normals, 
                view_dirs, 
                light_features, # 3 + 3 + 1 = 7 
                feature_vectors
            )
            diffuse_active_color = light_lum * light_net_color
            specular_active_color = 0

        elif self.bsdf_type == "ambient_sep_direct":
            # light_features = 1/(light_dir*light_dir).sum(-1,keepdim=True)
            # # light_features = (light_dir*light_dir).sum(-1,keepdim=True)
            light_features = (light_dir*light_dir).sum(-1,keepdim=True)
            light_net_color = self.light_net(
                points, 
                normals, 
                view_dirs, 
                light_features, # 3 + 3 + 1 = 7 
                feature_vectors
            )
            colocated, gi = torch.split(light_net_color, 3, dim=-1)
            diffuse_active_color = gi
            specular_active_color = 0
        elif self.bsdf_type == "iron":
            diffuse_active_color = self.ambient_net(
                                            points, 
                                            normals, 
                                            view_dirs, 
                                            brdf_params, 
                                            feature_vectors
                                        )
            specular_active_color = 0
        elif self.bsdf_type == "ambient_direct":
            diffuse_active_color = self.ambient_net(
                                            points, 
                                            normals, 
                                            view_dirs, 
                                            brdf_params, 
                                            feature_vectors
                                        )
            # light_features = torch.cat([light_origin.unsqueeze(dim=0).repeat(points.shape[0], 1), 1/(light_dir*light_dir).sum(-1,keepdim=True)], dim=1)
            # light_features = light_origin.unsqueeze(dim=0).repeat(points.shape[0], 1)
            # diffuse_active_color = self.light_net(
            #                                 points, 
            #                                 normals, 
            #                                 view_dirs, 
            #                                 light_features,
            #                                 feature_vectors
            #                             )
            
            # print(diffuse_active_color.min(), diffuse_active_color.max(), diffuse_active_color.mean(), self.flash_light_gamma())
            specular_active_color = 0
        else:
            raise NotImplementedError(self.bsdf_type)
        # print(diffuse_active_color)
        if self.is_darkroom: 
            ambient_color = torch.as_tensor(0).to(diffuse_active_color.device)
        else:
            if self.bsdf_type == "ambient":
                raise NotImplementedError()
            ambient_color = self.ambient_net(
                                            points, 
                                            normals, 
                                            view_dirs, 
                                            brdf_params, 
                                            feature_vectors 
                                        ) # do not use ambient to optimize geometry

            
            pass
        # if self.bsdf_type == "ambient_sep":
        #     ambient_color = light_net_color
        #     print("warning: ambient using light_net")
        # print("brdf_params", brdf_params, "diffuse_active_color", diffuse_active_color, "specular_active_color", specular_active_color, "flash_gamma", self.flash_light_gamma())
        #print("diffuse_flash_light", (diffuse_active_color) * self.flash_light_gamma())
        # print("flash_light_gamma", self.flash_light_gamma())
        #print("ambient_color", ambient_color, ambient_color.min(), ambient_color.max())
        physics = (diffuse_active_color + specular_active_color) * self.flash_light_gamma()
        #print("physicalbased", physics, physics.min(), physics.max())
        if self.bsdf_type=="iron":
            output = diffuse_active_color
        else:
            print("ambient color?", ambient_color, "diffuse_active_color", diffuse_active_color, "specular active color", specular_active_color, self.flash_light_gamma())
            output= ambient_color + (diffuse_active_color + specular_active_color) * self.flash_light_gamma()
        # print(diffuse_active_color, specular_active_color, self.flash_light_gamma(), brdf_params, light_lum)
        # print("ambient", ambient_net_color)
        # print("otuput", output)
        extra_out = {
            
        }
        if self.bsdf_type == "ambient_sep":
            extra_out['light_net_color'] = light_net_color
            pass
        extra_out["ambient_color"] = ambient_color
        return output, extra_out



# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class PhysicalNeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(PhysicalNeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 6)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views, light_origin, light_lum, return_density_only=False):

        if input_pts.shape[-1] == 4: # (..., 4)
            points = input_pts[..., :-1] / input_pts[..., -1:]
        else: # (..., 3)
            points = input_pts
        light_dir = points - light_origin
        irradiance = light_lum / (light_dir*light_dir).sum(-1,keepdim=True) # (..., 3) or (..., 1)

        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)
            

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            if return_density_only:
                return alpha, None
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb_environ, rgb_active = torch.split(self.rgb_linear(h), 3, dim=-1)
            # return alpha, safe_exp(rgb_environ) + safe_exp(rgb_active) * irradiance
            return alpha, rgb_environ + rgb_active * irradiance
        else:
            assert False
class PhysicalNeRFSep(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(PhysicalNeRFSep, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        embed_fn_light, input_ch_light = get_embedder(multires, input_dims=3)
        self.embed_fn_light = embed_fn_light
        self.input_ch_view += input_ch_light
        
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, output_ch)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views, light_origin, light_lum, return_density_only=False):

        if input_pts.shape[-1] == 4: # (..., 4)
            points = input_pts[..., :-1] / input_pts[..., -1:]
        else: # (..., 3)
            points = input_pts
        light_dir = points - light_origin
        # irradiance = light_lum / (light_dir*light_dir).sum(-1,keepdim=True) # (..., 3) or (..., 1)
        # print("input_pts", input_pts)
        # print("points", points)
        # print("light_origin", light_origin)
        # print("sq dist", (light_dir*light_dir).sum(-1,keepdim=True))
        # light_features = 1/(light_dir*light_dir).sum(-1,keepdim=True)
        light_features = light_origin.unsqueeze(dim=0).repeat(points.shape[0], 1)
        # print("light_features", light_features.min(), light_features.max(), light_features.mean())
        # inputs = torch.cat([input_pts, light_features], dim=-1)
        if self.embed_fn is not None:
            input_pts_embed = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)
            
        input_light = self.embed_fn_light(light_features)
        
        h = input_pts_embed
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts_embed, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            if return_density_only:
                return alpha, None
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views, input_light], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            rgb = safe_exp(rgb)
            # print("outside rgb", rgb.min(), rgb.max(), rgb.mean())
            # rgb[input_pts[:, 3]==1.0] = 0.0
            # print(rgb.shape)
            # print(input_pts[:,3:].shape)
            # rgb = torch.where(input_pts[:, 3:]==1.0, 0.0, rgb)
            # print("mask rate", (input_pts[:, 3:]==1.0).float().mean())
            # print("outside rgb after", rgb.min(), rgb.max(), rgb.mean())
            return alpha, rgb
        else:
            assert False

# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class PhysicalNeRFNGP(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires_view=0,
                 output_ch=4,
                 use_viewdirs=False):
        super(PhysicalNeRFNGP, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None
        self.embed_fn = tcnn.Encoding(
            n_input_dims=4,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            dtype=torch.float32
        )
        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view
            
        self.input_ch = 32
        self.use_viewdirs = use_viewdirs
        self.skips = []
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])
        
        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])
        
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 6)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views, light_origin, light_lum, return_density_only=False):

        if input_pts.shape[-1] == 4: # (..., 4)
            points = input_pts[..., :-1] / input_pts[..., -1:]
        else: # (..., 3)
            points = input_pts
        light_dir = points - light_origin
        irradiance = light_lum / (light_dir*light_dir).sum(-1,keepdim=True) # (..., 3) or (..., 1)
        
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)
            

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            if return_density_only:
                return alpha, None
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb_environ, rgb_active = torch.split(self.rgb_linear(h), 3, dim=-1)
            return alpha, rgb_environ + rgb_active * irradiance
        else:
            assert False
            
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views, light_origin, light_lum, return_density_only=False):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            if return_density_only:
                return alpha, None
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


# class MitsubaNeradWrapper(nn.Module):
#     def __init__(self, mitsuba_renderer):
#         self.mitsuba_renderer = mitsuba_renderer
        
#         pass

#     def forward(self,  points, normals, view_dirs, light_origin, light_lum, brdf_params, feature_vectors):

#         net_ins = self.extract_inputs(si, scene, sampler)
#         RHS_net = self.RHS_net(ray, scene, depth, sampler, active)
#         pass
