from typing import Any

import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.mitsuba_wrapper import MitsubaWrapper, wrapper_registry
from nerad.model.tcnn_embedding import TcnnEmbedding
from nerad.utils.mitsuba_utils import vec_to_tens_safe


from collections import defaultdict
from typing import Callable, Dict, Tuple, Optional
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nerad.utils.nerf_utils import sample_pdf
from nerad.model.config import NerfConfig

class NerfMlp(nn.Module):
    def __init__(self,
                 n_layers: int,
                 hidden_channels: int,
                 pts_channels: int,
                 dir_channels: int,
                 ):
        super().__init__()

        self.pts_linears = nn.ModuleList(
            [nn.Linear(pts_channels, hidden_channels)] +
            [nn.Linear(hidden_channels, hidden_channels) if i != 4 else
             nn.Linear(pts_channels + hidden_channels, hidden_channels) for i in range(n_layers - 1)]
        )
        self.dir_linear = nn.Linear(dir_channels + hidden_channels, hidden_channels // 2)
        self.feature_linear = nn.Linear(hidden_channels, hidden_channels)
        self.alpha_linear = nn.Linear(hidden_channels, 1)
        self.rgb_linear = nn.Linear(hidden_channels // 2, 3)

        self.pts_channels = pts_channels
        self.dir_channels = dir_channels
        self.hidden_channels = hidden_channels

    def forward(self, pts: Tensor, dirs: Tensor) -> Tuple[Tensor, ...]:
        h = pts
        for i in range(len(self.pts_linears)):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i == 4:
                h = torch.cat([pts, h], dim=-1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        h = torch.cat([feature, dirs], dim=-1)
        h = self.dir_linear(h)
        h = F.relu(h)

        rgb = self.rgb_linear(h)

        return rgb, alpha

class PositionalEncoding(nn.Module):
    def __init__(self,
                in_channels: int,
                n_freqs: int,
                max_freq_log2: int,
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels * (n_freqs * 2 + 1)
        self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, steps=n_freqs)

    def forward(self, x):
        return torch.cat([x] + list(chain.from_iterable((
            [torch.sin(x * freq), torch.cos(x * freq)] for freq in self.freq_bands
        ))), dim=-1)


class NerfModel(nn.Module):
    def __init__(self,
                 n_layers: int,
                 hidden_channels: int,
                 pts_embedding ,
                 dir_embedding,
                 coarse_net,
                 fine_net,
                 cfg: NerfConfig,
                 ):
        super().__init__()
        self.cfg = cfg

        self.pts_embedding = pts_embedding
        self.dir_embedding = dir_embedding

        self.coarse_net = coarse_net
        self.fine_net = fine_net

    def forward(self, rays_o: Tensor, rays_d: Tensor, view_dirs: Tensor, perturb: bool, raw_noise_std: float, mint, maxt) -> Dict[str, Tensor]:
        cfg = self.cfg
        device = rays_o.device

        n_rays = len(rays_o)
        n_samples = cfg.n_samples
        n_importance = cfg.n_importance

        t_vals = torch.linspace(0, 1, steps=n_samples)
        z_vals = mint.cpu()*(1-t_vals) + maxt.cpu()*(t_vals)
        z_vals = z_vals.expand([len(rays_o), n_samples])  # [N, S]

        if perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        z_vals = z_vals.to(device)
        pts = rays_o.view(n_rays, 1, 3) + rays_d.view(n_rays, 1, 3) * z_vals.view(n_rays, n_samples, 1)  # [N, S, 3]
        rgb, alpha = self.run_network(self.coarse_net, pts, view_dirs)
        rgb0, weights0, depths0 = self.convert_network_output(rgb, alpha, z_vals, rays_d, raw_noise_std)

        if cfg.n_importance == 0:
            return {
                "rgb": rgb0,
                "weights": weights0,
                "depths": depths0,
                "rgb0": rgb0,
                "weights0": weights0,
                "depths0": depths0,
            }

        # fine network

        z_samples = sample_pdf(
            0.5 * (z_vals[..., 1:] + z_vals[..., :-1]).cpu(),
            weights0.detach().cpu()[..., 1:-1],
            n_importance,
            det=not perturb,
        ).to(device)

        if cfg.importance_mode == "cat":
            z_vals = torch.cat([z_vals, z_samples], dim=-1)
        else:
            z_vals = z_samples

        z_vals, _ = torch.sort(z_vals, dim=-1)
        pts = rays_o.view(n_rays, 1, 3) + rays_d.view(n_rays, 1, 3) * z_vals.view(n_rays, -1, 1)

        rgb, alpha = self.run_network(self.fine_net, pts, view_dirs)
        rgb, weights, depths = self.convert_network_output(rgb, alpha, z_vals, rays_d, raw_noise_std)

        return {
            "rgb": rgb,
            "weights": weights,
            "depths": depths,
            "rgb0": rgb0,
            "weights0": weights0,
            "depths0": depths0,
        }

    def run_network(self, network: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]], pts: Tensor, view_dirs: Tensor):
        """run network by expanding view_dirs

        pts: [N, S, 3]
        view_dirs: [N, 3]
        """

        shape = list(pts.shape)
        pts = pts.view(-1, 3)
        view_dirs = view_dirs.view(-1, 1, 3).expand(shape).reshape(-1, 3)

        rgb, alpha = network(
            self.pts_embedding(pts),
            self.dir_embedding(view_dirs),
        )
        rgb = rgb.view(shape)  # [N, S, 3]
        alpha = alpha.view(shape[:-1])  # [N, S]

        return rgb, alpha

    def run_my_network(self, network: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]], pts: Tensor, view_dirs: Tensor):
        """run network by expanding view_dirs

        pts: [N, S, 3]
        view_dirs: [N, 3]
        """

        pts = pts.view(-1, 3)
        view_dirs = view_dirs.view(-1, 3)

        rgb, alpha = network(
            self.pts_embedding(pts),
            self.dir_embedding(view_dirs),
        )
        return rgb, alpha


    def render_rays(self, rays_o: Tensor, rays_d: Tensor, mint, maxt) -> Dict[str, Tensor]:
        cfg = self.cfg
        shape = list(rays_o.shape)  # [*, 3]
        shape[-1] = -1  # [*, -1]

        rays_o = rays_o.view([-1, 3])
        rays_d = rays_d.view([-1, 3])
        view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        output = defaultdict(list)

        output = defaultdict(list)
        chunk_size = cfg.ray_chunk_size
        for i in range(0, len(rays_o), chunk_size):
            chunk_output = self.forward(
                rays_o[i:i+chunk_size],
                rays_d[i:i+chunk_size],
                view_dirs[i:i+chunk_size],
                False,
                0,
                mint[i:i+chunk_size],
                maxt[i:i+chunk_size],
            )
            for k, v in chunk_output.items():
                output[k].append(v)

        return {k: torch.cat(v, dim=0).view(shape).squeeze(-1) for k, v in output.items()}


    def convert_network_output(self, rgb: Tensor, alpha: Tensor, z_vals: Tensor, rays_d: Tensor, raw_noise_std: float) -> Tuple[Tensor, ...]:
        n_rays = len(rays_d)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)], dim=-1)  # [N, S]
        dists = dists * torch.norm(rays_d.view(n_rays, 1, 3), dim=-1)

        rgb = torch.sigmoid(rgb)
        noise = 0
        if raw_noise_std > 0:
            noise = torch.randn(alpha.shape) * raw_noise_std

        alpha = 1 - torch.exp(-F.relu(alpha + noise) * dists)
        weights = alpha * torch.cumprod(torch.cat([
            torch.ones((alpha.shape[0], 1), device=alpha.device),
            1-alpha + 1e-10
        ], -1), -1)[:, :-1]
        rgb = torch.sum(weights[..., None] * rgb, dim=-2)
        depths = torch.sum(weights * z_vals, -1)
        acc = torch.sum(weights, dim=-1)

        if self.cfg.white_background:
            rgb = rgb + (1 - acc[..., None])

        return rgb, weights, depths


@wrapper_registry.register("nerf_net")
class MitsubaNerfNetworkWrapper(MitsubaWrapper):
    def __init__(
        self,
        n_layers: int,
        hidden_channels: int,
        embedding: dict[str, Any],
        config: dict[str, Any],
        scene_min: Any,
        scene_max: Any,
    ):
        super().__init__(scene_min = 0, scene_max = 1, name = "nerf_net")
        pts_embedding = PositionalEncoding(3, embedding["pos_enc_freq"], embedding["pos_enc_freq"]-1)
        dir_embedding = PositionalEncoding(3, embedding["dir_enc_freq"], embedding["dir_enc_freq"]-1)

        pts_channels = pts_embedding.out_channels
        dir_channels = dir_embedding.out_channels

        coarse_net = NerfMlp(n_layers, hidden_channels, pts_channels, dir_channels)
        fine_net = None
        if config["n_importance"]>0:
            fine_net = NerfMlp(n_layers, hidden_channels, pts_channels, dir_channels)

        self.network = NerfModel(n_layers, hidden_channels, pts_embedding, dir_embedding, coarse_net, fine_net, NerfConfig(**config))

    def _eval(self, pts, dirs, norms, albedo):
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        d_tensor = vec_to_tens_safe(dirs + self.grad_activator)
        rgb, rgb0 = self.eval_torch(
            p_tensor, d_tensor, norms, albedo)

        rgb = dr.unravel(mi.Vector3f, rgb.array)
        rgb0 = dr.unravel(mi.Vector3f, rgb0.array)

        return rgb, rgb0


    def _eval_out(self, pts, dirs, norms, albedo):
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        d_tensor = vec_to_tens_safe(dirs + self.grad_activator)
        rgb, alpha = self.eval_network_alpha(
            p_tensor, d_tensor, norms, albedo)

        rgb = dr.unravel(mi.Vector3f, rgb.array)
        alpha = mi.Float(alpha.array)

        return rgb, alpha


    @dr.wrap_ad(source='drjit', target='torch')
    def eval_torch(self, pts, dirs, mint, maxt):
        output = self.network.render_rays(pts, dirs, mint, maxt)
        return output['rgb'], output['rgb0']

    @dr.wrap_ad(source='drjit', target='torch')
    def eval_network_alpha(self, pts, dirs, mint, maxt):
        rgb, alpha = self.network.run_my_network(self.network.coarse_net, pts, dirs)
        return rgb, alpha


    def _traverse(self, callback):
        callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)
