from typing import Any

import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.mitsuba_wrapper import MitsubaWrapper, wrapper_registry
from nerad.model.tcnn_embedding import TcnnEmbedding
from nerad.utils.mitsuba_utils import vec_to_tens_safe
import numpy as np

class ReflectanceMlp(nn.Module):
    def __init__(
        self,
        width: int,
        hidden: int,
        embedding: dict[str, Any],
        use_sigmoid: bool=True
    ):
        super().__init__()

        self.embedding = TcnnEmbedding(embedding)
        in_size = 3 + self.embedding.n_output_dims

        hidden_layers = []
        for _ in range(hidden):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.LeakyReLU(inplace=True))
        layers = [
            nn.Linear(in_size, width),
            nn.LeakyReLU(inplace=True),
            *hidden_layers,
            nn.Linear(width, 3),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.network = nn.Sequential(
            *layers
        )

    def forward(self, points):
        net_in = torch.cat([points, self.embedding(points)], dim=-1)
        ret = self.network(net_in)
        return ret

@wrapper_registry.register("reflectance_net")
class MitsubaReflectanceNetworkWrapper(MitsubaWrapper):
    def __init__(
        self,
        width: int,
        hidden: int,
        embedding: dict[str, Any],
        scene_min: Any,
        scene_max: Any,
        clamp: bool=True,
        scale: float=1.0,
        offset:float=0.0
    ):
        super().__init__(scene_min, scene_max, "bsdf_net")
        self.network = ReflectanceMlp(width, hidden, embedding, use_sigmoid=clamp)
        self.clamp=clamp
        self.scale = scale
        self.offset = offset
        self.repeat_pattern = None
        self.last_cache = None
        self.last_pts = None
        self.override_output = None

    def _eval(self, pts, dirs, norms, albedo, pts2, dirs2, em_weight, active=True):
        pts = 2 * pts - 1
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        torch_out = self.eval_torch(p_tensor, active)
        dr.make_opaque(torch_out)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        result = output
        if self.clamp:
            result = dr.clamp(output, 0, 1)
        result *= self.scale
        result += self.offset
        return result

    def _eval_and_grad(self, pts, active=True):
        pts = 2 * pts - 1
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        torch_out, torch_out_grad = self.eval_torch_and_grad(p_tensor, active)
        dr.make_opaque(torch_out)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        output_grad = dr.unravel(mi.Vector3f, torch_out_grad.array)

        result = output
        if self.clamp:
            result = dr.clamp(output, 0, 1)
        result *= self.scale
        result += self.offset

        output_grad *= self.scale

        return result, output_grad

    def set_repeat_pattern_hint(self, repat_pattern):
        self.repeat_pattern = repat_pattern

    def set_override_output(self, override_output):
        self.override_output = override_output

    def eval_torch_(self, pts, active):
        if self.override_output is not None:
            print("WARNING reflectance net: using override output")
            return self.override_output
        if isinstance(active, mi.Bool):

           active = torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device)
        else:
            active = active.unsqueeze(dim=-1).to(pts.device)
        assert torch.isfinite(torch.where(active, pts, 0.0)).all()
        # matched = False
        # if self.repeat_pattern is not None and (pts.shape[0] % self.repeat_pattern) == 0:

        #     pts_test = pts.reshape(-1, self.repeat_pattern, 3)
        #     # print('pattern match', (pts_test[:, 0:1, :] - pts_test == 0).all())
        #     if (pts_test[:, 0:1, :] - pts_test == 0).all(): # we matched the pattern
        #         matched = True
        #         pts = pts_test[:, 0, :]
        #         # print(active)
        #         active = active.reshape(-1, self.repeat_pattern, 1)[:, 0]

        result = self.network(torch.where(active, pts, 0.0))
        # print("in reflectance", result)
        # if self.repeat_pattern is not None and matched:
        #     pts = torch.repeat_interleave(pts, self.repeat_pattern, dim=0)
        #     result = torch.repeat_interleave(result, self.repeat_pattern, dim=0)
        # self.last_cache, self.last_pts = result, pts
        # print("in reflectance result", result)
        return result
    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, pts, active):
        return self.eval_torch_(pts, active)

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch_and_grad(self, pts, active):
        result = self.eval_torch_(pts, active)
        mean = result.sum()
        grad = torch.autograd.grad(mean, pts, create_graph=True)[0]
        return result, grad
    def _traverse(self, callback):
        callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)


@wrapper_registry.register("fake_reflectance_net")
class FakeMitsubaReflectanceNetworkWrapper(MitsubaReflectanceNetworkWrapper):
    def __init__(
        self,
        width: int,
        hidden: int,
        embedding: dict[str, Any],
        scene_min: Any,
        scene_max: Any,
        value: float,
    ):
        super().__init__(width, hidden, embedding, scene_min, scene_max)
        self.value = value

    def _eval(self, net_ins):
        output = super()._eval(net_ins)
        return dr.clamp(self.value + 0.0001 * output, 0, 1)
