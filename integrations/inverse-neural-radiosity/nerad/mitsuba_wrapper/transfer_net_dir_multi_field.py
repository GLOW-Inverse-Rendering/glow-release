from typing import Any

import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.mitsuba_wrapper import MitsubaWrapper, wrapper_registry
from nerad.utils.mitsuba_utils import vec_to_tens_safe
from nerad.utils.embedding_utils import create_embedding, embed

import numpy as np
class TransferMLP(nn.Module):
    def __init__(
        self,
        width: int,
        hidden: int,
        position1_embedding: dict[str, Any],
        direction_embedding: dict[str, Any],
        position2_embedding: dict[str, Any],  #In photometric sterio case, this is the point light location
        direction2_embedding: dict[str, Any],  #In photometric sterio case, this is the point light location
    ):
        super().__init__()
        self.pos_emb = create_embedding(position1_embedding)
        self.pos2_emb = create_embedding(position2_embedding)
        self.dir_emb = create_embedding(direction_embedding)
        self.dir2_emb = create_embedding(direction2_embedding)

        self.norm_emb = create_embedding(direction_embedding)

        def embed_size(in_vector, embedding):
            return embed(torch.zeros(1, in_vector).cuda(), embedding).shape[-1]

        #input size : points + direction + points2
        in_size = embed_size(3, self.pos_emb) + embed_size(3, self.dir_emb) + embed_size(3, self.pos2_emb) + embed_size(3, self.dir2_emb)

        in_size += embed_size(3, self.norm_emb)

        # print("total in size", in_size)
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
        self.input_scale = None
    def set_input_scale(self, scale):
        self.input_scale = scale
        pass

    def forward(self, points, dirs, normals, albedo, points2, dirs2):
        if self.input_scale is not None:
            points = points * self.input_scale
            points2 = points2 * self.input_scale
        net_in = torch.cat(
            [
                embed(points, self.pos_emb),
                embed(dirs, self.dir_emb),
                embed(points2, self.pos2_emb),
                embed(dirs2, self.dir2_emb),

            ],
            dim=-1,
        )
        net_in = torch.cat([
            net_in,
            embed(normals, self.norm_emb),
        ], dim=-1)
        ret = self.network(net_in)
        return torch.abs(ret)

@wrapper_registry.register("transfer_net_dir_multi_field")
class MitsubaTransferNetworkWrapper(MitsubaWrapper):
    def __init__(
        self,
        width: int,
        hidden: int,
        position1_embedding: dict[str, Any],
        direction_embedding: dict[str, Any],
        position2_embedding: dict[str, Any],
        direction2_embedding,
        scene_min: Any,
        scene_max: Any,
        num_fields: int
    ):
        super().__init__(scene_min, scene_max, "transfer_net_dir_multi_field")
        self.transfer_mlp_factory = lambda: TransferMLP(width, hidden, position1_embedding, direction_embedding, position2_embedding, direction2_embedding)
        self.network = nn.ModuleList([
            self.transfer_mlp_factory()
            for _ in range(num_fields)

        ])
        self.num_fields = num_fields
        self.args = (width, hidden, position1_embedding, direction_embedding, position2_embedding, direction2_embedding, scene_min, scene_max, num_fields)

    def clone(self):
        net = MitsubaTransferNetworkWrapper(*self.args)
        net.network.load_state_dict(self.network.state_dict())
        return net

    def reset_field(self, idx):
        self.network[idx] = self.transfer_mlp_factory()


    def _eval(self, pts, dirs, norms, albedo, pts2, dirs2, em_weight, active):
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        d_tensor = vec_to_tens_safe(dirs)
        n_tensor = vec_to_tens_safe(norms) if norms is not None else None
        alb_tensor = vec_to_tens_safe(albedo) if albedo is not None else None
        p2_tensor = vec_to_tens_safe(pts2)
        d2_tensor = vec_to_tens_safe(dirs2)
        em_tensor = vec_to_tens_safe(em_weight)
        torch_out = self.eval_torch(
            p_tensor, d_tensor, n_tensor, alb_tensor, p2_tensor, d2_tensor, em_tensor, active)
        dr.make_opaque(torch_out)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        return dr.abs(output)

    @dr.wrap_ad(source='drjit', target='torch')
    def eval_torch(self, pts, dirs, norms, albedo, p2_tensor, d2_tensor, em_tensor, active):
        em_tensor = em_tensor[:, 0]
        active = torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device)
        pts = torch.where(active, pts, 0.0)
        dirs = torch.where(active, dirs, 0.0)
        norms = torch.where(active, norms, 0.0) if norms is not None else None
        albedo = torch.where(active, albedo, 0.0) if albedo is not None else None
        p2_tensor = torch.where(active, p2_tensor, 0.0)
        d2_tensor = torch.where(active, d2_tensor, 0.0)
        em_tensor = torch.where(active.squeeze(), em_tensor, 0).long()
        out = torch.zeros_like(pts)
        # em == 0: no occ, em == 1 occ, em == 2: black, em == 3: network 0 only
        net_direct_out  = self.network[0](pts, dirs, norms, albedo, p2_tensor, d2_tensor)
        net_indirect_out  = self.network[1](pts, dirs, norms, albedo, p2_tensor, d2_tensor)
        mask_0 = em_tensor == 0 # no occ
        out[mask_0] = net_direct_out[mask_0] + net_indirect_out[mask_0]

        mask_1 = em_tensor == 1 #  occ
        out[mask_1] = net_indirect_out[mask_1]

        mask_3 = em_tensor == 3 # network 0 only
        out[mask_3] = net_direct_out[mask_3]

        mask_2 = em_tensor == 2 #  occ
        if self.num_fields == 3: # ambient light
            mask_4 = em_tensor == 4
            net_ambient_out  = self.network[2](pts, dirs, norms, albedo, p2_tensor, d2_tensor)
            out[mask_4] = net_ambient_out[mask_4]
        return out

    def _traverse(self, callback):
        callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)
