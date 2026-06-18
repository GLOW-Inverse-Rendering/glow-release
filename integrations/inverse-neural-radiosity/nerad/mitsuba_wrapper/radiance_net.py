from typing import Any

import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.mitsuba_wrapper import MitsubaWrapper, wrapper_registry
from nerad.utils.mitsuba_utils import vec_to_tens_safe
from nerad.utils.embedding_utils import create_embedding, embed


class RadianceMLP(nn.Module):
    def __init__(
        self,
        width: int,
        hidden: int,
        position_embedding: dict[str, Any],
        direction_embedding: dict[str, Any],
        scene_properties_input: bool
    ):
        super().__init__()
        self.scene_properties_input = scene_properties_input
        self.pos_emb = create_embedding(position_embedding)
        self.dir_emb = create_embedding(direction_embedding)

        def embed_size(in_vector, embedding):
            return embed(torch.zeros(1, in_vector).cuda(), embedding).shape[-1]

        #input size : points + direction
        in_size = embed_size(3, self.pos_emb) + embed_size(3, self.dir_emb)

        if scene_properties_input:
            in_size += embed_size(3, self.dir_emb)      #normal
            in_size += 3                                #albedo

        hidden_layers = []
        for _ in range(hidden):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(
            nn.Linear(in_size, width),
            nn.ReLU(inplace=True),
            *hidden_layers,
            nn.Linear(width, 3),
        )

    def forward(self, points, dirs, normals, albedo):
        net_in = torch.cat(
            [
                embed(points, self.pos_emb),
                embed(dirs, self.dir_emb)
            ],
            dim=-1,
        )
        if self.scene_properties_input:
            net_in = torch.cat(
                [
                    net_in,
                    embed(normals, self.dir_emb),
                    albedo],
                dim=-1,
            )

        ret = self.network(net_in)
        return torch.abs(ret)

@wrapper_registry.register("radiance_net")
class MitsubaRadianceNetworkWrapper(MitsubaWrapper):
    def __init__(
        self,
        width: int,
        hidden: int,
        position_embedding: dict[str, Any],
        direction_embedding: dict[str, Any],
        scene_min: Any,
        scene_max: Any,
        scene_properties_input,
    ):
        super().__init__(scene_min, scene_max, "radiance_net")
        self.network = RadianceMLP(width, hidden, position_embedding, direction_embedding, scene_properties_input)

    def _eval(self, pts, dirs, norms, albedo, pts2, dirs2, em_weight, active=True):
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        d_tensor = vec_to_tens_safe(dirs)
        n_tensor = vec_to_tens_safe(norms)
        alb_tensor = vec_to_tens_safe(albedo)
        torch_out = self.eval_torch(
            p_tensor, d_tensor, n_tensor, alb_tensor)

        output = dr.unravel(mi.Vector3f, torch_out.array)
        return dr.abs(output)

    @dr.wrap_ad(source='drjit', target='torch')
    def eval_torch(self, pts, dirs, norms, albedo):
        return self.network(pts, dirs, norms, albedo)

    def _traverse(self, callback):
        callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)
