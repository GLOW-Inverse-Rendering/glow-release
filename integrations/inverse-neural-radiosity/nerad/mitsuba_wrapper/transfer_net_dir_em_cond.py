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
        scene_properties_input: bool,
        no_albedo: bool = False,
        em_cond = False
    ):
        super().__init__()
        self.args = (width, hidden, position1_embedding, direction_embedding, position2_embedding, direction2_embedding, scene_properties_input, no_albedo, em_cond)
        self.scene_properties_input = scene_properties_input
        self.no_albedo = no_albedo
        self.em_cond = em_cond
        self.pos_emb = create_embedding(position1_embedding)
        self.pos2_emb = create_embedding(position2_embedding)
        self.dir_emb = create_embedding(direction_embedding)
        self.dir2_emb = create_embedding(direction2_embedding)

        def embed_size(in_vector, embedding):
            return embed(torch.zeros(1, in_vector).cuda(), embedding).shape[-1]

        #input size : points + direction + points2
        in_size = embed_size(3, self.pos_emb) + embed_size(3, self.dir_emb) + embed_size(3, self.pos2_emb) + embed_size(3, self.dir2_emb)

        if scene_properties_input:
            in_size += embed_size(3, self.dir_emb)      #normal
            if not self.no_albedo:
                in_size += 3                                #albedo
        if self.em_cond:
            in_size += 3

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


    def forward(self, points, dirs, normals, albedo, points2, dirs2, em_weight):
        net_in = torch.cat(
            [
                embed(points, self.pos_emb),
                embed(dirs, self.dir_emb),
                embed(points2, self.pos2_emb),
                embed(dirs2, self.dir2_emb),

            ],
            dim=-1,
        )
        if self.scene_properties_input:
            if not self.no_albedo:
                net_in = torch.cat(
                    [
                        net_in,
                        embed(normals, self.dir_emb),
                        albedo],
                    dim=-1,
                )
            else:
                net_in = torch.cat(
                    [
                        net_in,
                        embed(normals, self.dir_emb),
                        ],
                    dim=-1,
                )
        if self.em_cond:
            net_in = torch.cat(
                [
                    net_in,
                    em_weight
                    ],
                dim=-1,
            )
        ret = self.network(net_in)
        return torch.abs(ret)

@wrapper_registry.register("transfer_net_dir_em_cond")
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
        scene_properties_input,
        no_albedo=False,
        em_cond=False
    ):
        super().__init__(scene_min, scene_max, "transfer_net_dir_em_cond")
        self.network = TransferMLP(width, hidden, position1_embedding, direction_embedding, position2_embedding, direction2_embedding, scene_properties_input, no_albedo, em_cond)
        self.args = (width, hidden, position1_embedding, direction_embedding, position2_embedding, direction2_embedding, scene_min, scene_max, scene_properties_input, no_albedo, em_cond)
    def clone(self):
        net = MitsubaTransferNetworkWrapper(*self.args)
        net.network.load_state_dict(self.network.state_dict())
        return net
    def _eval(self, pts, dirs, norms, albedo, pts2, dirs2, em_weight, active):
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        # print("ref, pts", dr.grad_enabled(pts))
        # print("ref, grad_activator", dr.grad_enabled(self.grad_activator))
        # print("ref, p_tensor", dr.grad_enabled(p_tensor))

        d_tensor = vec_to_tens_safe(dirs)
        n_tensor = vec_to_tens_safe(norms) if norms is not None else None
        # print(albedo)
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
        pts = torch.where(torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device), pts, 0.0)
        dirs = torch.where(torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device), dirs, 0.0)
        norms = torch.where(torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(norms.device), norms, 0.0) if norms is not None else None
        albedo = torch.where(torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device), albedo, 0.0) if albedo is not None else None
        p2_tensor = torch.where(torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device), p2_tensor, 0.0)
        d2_tensor = torch.where(torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device), d2_tensor, 0.0)
        em_tensor = torch.where(torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device), em_tensor, 0.0)
        # print(np.array(active))
        # print("pts", pts)
        # print("pts", torch.isfinite(pts).all())
        # print("dirs", torch.isfinite(dirs).all())
        # print("norms", torch.isfinite(norms).all())
        # print("albedo", torch.isfinite(albedo).all())
        # print("p2_tensor", torch.isfinite(p2_tensor).all())
        return self.network(pts, dirs, norms, albedo, p2_tensor, d2_tensor, em_tensor)

    def _traverse(self, callback):
        callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)
