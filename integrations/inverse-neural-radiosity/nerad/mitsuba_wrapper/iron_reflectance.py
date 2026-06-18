from nerad.model.wildlight_model import RenderingNetworkIRON as RenderingNetwork
from nerad.model.wildlight_embedder import get_embedder

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
class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
    ):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[..., :1] / self.scale, x[..., 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)
    @torch.enable_grad()
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients

    def get_all(self, x, is_training=True):
        with torch.enable_grad():
            x.requires_grad_(True)
            tmp = self.forward(x)
            y, feature = tmp[..., :1], tmp[..., 1:]
            # print("y", y)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=is_training,
                retain_graph=is_training,
                only_inputs=True,
            )[0]
        if not is_training:
            return y.detach(), feature.detach(), gradients.detach()
        return y, feature, gradients

class ReflectanceMlpIron(nn.Module):
    def __init__(self, init_ckpt, type):
        super().__init__()
        self.sdf_network = SDFNetwork(
            d_in=3,
            d_out=257,
            d_hidden=256,
            n_layers=8,
            skip_in=[
                4,
            ],
            multires=6,
            bias=0.5,
            scale=1.0,
            geometric_init=True,
            weight_norm=True,
        ).cuda() #FIXME

        self.color_network_dict = {
            "diffuse_albedo_network": RenderingNetwork(
                d_in=9,
                d_out=3,
                d_feature=256,
                d_hidden=256,
                n_layers=4,
                multires_view=4,
                weight_norm=True,
                mode="idr",
                squeeze_out=True,
                # flip_rgb=True
            ).cuda(),
            "specular_albedo_network": RenderingNetwork(
                d_in=6,
                d_out=3,
                d_feature=256,
                d_hidden=256,
                n_layers=4,
                multires=6,
                multires_view=-1,
                mode="no_view_dir",
                squeeze_out=False,
                output_bias=0.4,
                output_scale=0.1,
            ).cuda(),
            "specular_roughness_network": RenderingNetwork(
                d_in=6,
                d_out=1,
                d_feature=256,
                d_hidden=256,
                n_layers=4,
                multires=6,
                multires_view=-1,
                mode="no_view_dir",
                squeeze_out=True,
                output_bias=0.1,
                output_scale=0.1,
            ).cuda(),
            # "point_light_network": PointLightNetwork().cuda(),
        }
        ckpt = torch.load(init_ckpt, map_location=torch.device("cuda"))
        print(ckpt.keys())
        self.sdf_network.load_state_dict(ckpt["sdf_network"])
        for x in list(self.color_network_dict.keys()):
            self.color_network_dict[x].load_state_dict(ckpt[x])
            pass
        self.type = type
        assert self.type in ["albedo", "roughness"]

    def forward(self, points):
        sdfs, features, normals = self.sdf_network.get_all(points)
        diffuse_albedo = self.color_network_dict["diffuse_albedo_network"](points, normals, -normals, features).abs()[
            ..., [2, 1, 0]
        ]

        specular_roughness = self.color_network_dict["specular_roughness_network"](points, normals, None, features).abs() + 0.01
        specular_roughness = specular_roughness.repeat(1, 3)
        # print(points.shape)
        # print(specular_roughness.shape)
        if self.type == "albedo":
            return diffuse_albedo
        elif self.type == "roughness":
            return specular_roughness
        else:
            raise NotImplementedError(self.type)

@wrapper_registry.register("iron_reflectance_net")
class MitsubaReflectanceNetworkIronWrapper(MitsubaWrapper):
    def __init__(
        self,
        init_ckpt,
        type,
        scene_min: Any,
        scene_max: Any,
    ):
        super().__init__(scene_min, scene_max, "bsdf_net")
        self.network = ReflectanceMlpIron(init_ckpt, type)

    def eval(self, pts, dirs=None, norms=None, albedo=None, pts2=None, active=True):
        # print(pts)
        result = self._eval(pts, dirs, norms, albedo, pts2, active)
        return result

    def _eval(self, pts, dirs, norms, albedo, pts2, active=True):
        # assert self.sdf is not None
        p_tensor = vec_to_tens_safe(pts + self.grad_activator)
        # with dr.suspend_grad(when=not self.optimize_geometry):
        #     with torch.set_grad_enabled(self.optimize_geometry):
        #         torch_features = self.eval_features(p_tensor, active)

        torch_out = self.eval_torch(p_tensor)
        dr.make_opaque(torch_out)
        output = dr.unravel(mi.Vector3f, torch_out.array)
        result = dr.clamp(output, 0, 1)
        return result
    # @dr.wrap_ad(source="drjit", target="torch")
    # def eval_features(self, pts, active):
    #     active = torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device)
    #     active2 = torch.isfinite(pts)
    #     active_all = active & active2
    #     inp = torch.where(active_all, pts, 0.0)
    #     # inp = torch.where(active, pts, 0.0)
    #     features = self.sdf[0].eval_feature_torch(inp)
    #     return features

    @dr.wrap_ad(source="drjit", target="torch")
    def eval_torch(self, points):
        # print(np.array(active).shape)
        # print(pts.shape)
        # print("torch finite pts", torch.isfinite(torch.where(torch.from_numpy(np.array(active)).unsqueeze(dim=-1).to(pts.device), pts, 0.0)).all())
        # feature_vector = self.sdf
        return self.network(points)

    def _traverse(self, callback):
        # callback.put_parameter("network", self.network, mi.ParamFlags.Differentiable)
        # callback.put_parameter("feat.grad_activator", self.feat_grad_activator, mi.ParamFlags.Differentiable)
        pass
