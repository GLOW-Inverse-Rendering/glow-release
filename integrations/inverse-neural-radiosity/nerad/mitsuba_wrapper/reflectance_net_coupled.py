from typing import Any

import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.mitsuba_wrapper import MitsubaWrapper, wrapper_registry
from nerad.model.tcnn_embedding import TcnnEmbedding
from nerad.utils.mitsuba_utils import vec_to_tens_safe
import numpy as np
from mytorch.utils.profiling_utils import counter_profiler, time_profiler

class ReflectanceMlpCoupled(nn.Module):
    def __init__(
        self,
        width: int,
        hidden: int,
        feature_size: int
    ):
        super().__init__()
        in_size = feature_size

        hidden_layers = []
        for _ in range(hidden):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.LeakyReLU(inplace=True))

        self.network = nn.Sequential(
            nn.Linear(in_size, width),
            nn.LeakyReLU(inplace=True),
            *hidden_layers,
            nn.Linear(width, 3),
            nn.Sigmoid()
        )

    def forward(self, features):
        net_in = features
        ret = self.network(net_in)

        return ret
@wrapper_registry.register("reflectance_net_coupled_external_net")
class MitsubaReflectanceNetworkCoupledExternalWrapper(MitsubaWrapper):
    def __init__(
        self,
        network,
        sdf, # sdf can be none
        feature_size,
        scene_min: Any,
        scene_max: Any,
    ):
        super().__init__(scene_min, scene_max, "bsdf_net")
        self.network = network
        self.optimize_geometry = False
        self.feat_grad_activator = mi.TensorXf(0.0, [1, feature_size])
        self.sdf = sdf
    def eval(self, pts, dirs=None, norms=None, albedo=None, pts2=None, active=True):
        if counter_profiler.enabled:
            counter_profiler.record(f"{self.name}.eval.pts", dr.shape(pts)[1])
        time_profiler.start(f"{self.name}.eval")
        result = self._eval(pts, dirs, norms, albedo, pts2, active)
        time_profiler.end(f"{self.name}.eval")
        return result

    def _eval(self, pts, dirs, norms, albedo, pts2, active=True):
        assert self.sdf is not None
        # pts = 2 * pts - 1
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        with dr.suspend_grad(when=not self.optimize_geometry):
            with torch.set_grad_enabled(self.optimize_geometry):
                torch_features = self.eval_features(p_tensor, active)

        torch_out = self.eval_torch(torch_features + self.feat_grad_activator)
        dr.make_opaque(torch_out)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        result = dr.clamp(output, 0, 1)
        return result
    @dr.wrap_ad(source="drjit", target="torch")
    def eval_features(self, pts, active):
        active = torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device)
        active2 = torch.isfinite(pts)
        active_all = active & active2
        inp = torch.where(active_all, pts, 0.0)
        features = self.sdf[0].eval_feature_torch(inp)
        return features

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, features):
        assert self.sdf is not None
        return self.network(features)

    def _traverse(self, callback):
        callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)
        callback.put_parameter("feat.grad_activator", self.feat_grad_activator, mi.ParamFlags.Differentiable)

@wrapper_registry.register("reflectance_net_coupled")
class MitsubaReflectanceNetworkCoupledWrapper(MitsubaReflectanceNetworkCoupledExternalWrapper):
    def __init__(
        self,
        width: int,
        hidden: int,
        feature_size: int,
        scene_min: Any,
        scene_max: Any,
    ):

        network = ReflectanceMlpCoupled(width, hidden, feature_size)
        super().__init__(network, None, feature_size, scene_min, scene_max) # sdf is none and require initialization in code
