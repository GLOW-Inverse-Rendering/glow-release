import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
import tinycudann as tcnn
import math
# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 scale_input_only=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()
        print("inside_outside", inside_outside)
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        self.scale_input_only = scale_input_only

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale_input_only
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            # print(x.device)
            # print(lin.weight.device)
            x = lin(x)
            
            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    @torch.enable_grad()
    def sdf_and_gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # print("sdf_and_grad requires_grad", y.requires_grad, gradients.requires_grad)
        return y, gradients.unsqueeze(1)

    @torch.enable_grad()
    def gradient(self, x):
        y,g = self.sdf_and_gradient(x)
        return g
    
    @torch.enable_grad()
    def hessian(self, x): 
        y,g,h=self.eval_all(x)
        return h
    
    @torch.enable_grad()
    def eval_all(self, x): 
        x.requires_grad_(True)
        y = self(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0] # B x 3
        hess = torch.empty(x.shape[0], 3, 3, dtype=x.dtype, device=x.device)
        for i in range(3):
            # print("i", i)
            hess[:, i, :] = torch.autograd.grad(
                outputs=gradients[:, i:i+1],
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
            )[0]
        # print("all requires_grad", y.requires_grad, gradients.requires_grad, hess.requires_grad)
        return y, gradients, hess

class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers, scale, bias, geometric_init, inside_outside):
        super().__init__()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        # torch.set_printoptions(threshold=10_000)
        for l in range(0, self.num_layers-1):
            out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                # print("before init", lin.weight)
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                # elif multires > 0 and l in self.skip_in:
                #     torch.nn.init.constant_(lin.bias, 0.0)
                #     torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                #     torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                # print("after init", lin.weight)
            # if weight_norm:
            #     lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
            self.activation = nn.ReLU(inplace=True)
            pass
        pass
    
    def forward(self, x):
        # BEGIN orignal fc code

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            # if l in self.skip_in:
            #     x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x
        # END orignal fc code


class TCNNMLP(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers, bias=0.5, scale=1, geometric_init=True, inside_outside=False):
        super().__init__()
        self.mlp = tcnn.Network(
            n_input_dims=d_in,
            n_output_dims=d_out,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": d_hidden,
                "n_hidden_layers": n_layers,
            },
        )
        fused_params = list(self.mlp.parameters())
        assert len(fused_params) == 1, len(fused_params)
        fused_params = fused_params[0]
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        offset = 0
        # self.bias = torch.nn.Parameter(torch.zeros(d_out, dtype=self.mlp.dtype))
        self.bias = -bias if not inside_outside else bias

        self.geometric_init = geometric_init
        pad_16_f = lambda x: math.ceil(x/16)*16
        # torch.set_printoptions(threshold=10_000)
        if geometric_init:
            for l in range(0, self.num_layers - 1):
                in_dim = dims[l]
                out_dim = dims[l+1]
                in_dim_ = pad_16_f(in_dim)
                out_dim_ = pad_16_f(out_dim)
                elements = in_dim_ * out_dim_
                # print("before init", fused_params[offset:offset+elements])
                if l == self.num_layers - 2:                    
                    if not inside_outside:
                        torch.nn.init.normal_(fused_params[offset:offset+elements], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        pass
                    else:
                        torch.nn.init.normal_(fused_params[offset:offset+elements], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        pass
                    # if not inside_outside:
                    #     torch.nn.init.normal_(fused_params[offset:, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    #     torch.nn.init.constant_(lin.bias, -bias)
                    # else:
                    #     torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    #     torch.nn.init.constant_(lin.bias, bias)
                elif l == 0:
                    torch.nn.init.constant_(fused_params[offset:offset+elements], 0.0)
                    # torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    for i in range(out_dim_):
                        torch.nn.init.normal_(fused_params[offset+i*in_dim_:offset+i*in_dim_+3], mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim))
                        pass
                else:
                    torch.nn.init.normal_(fused_params[offset:offset+elements], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    pass
                # print("post init", fused_params[offset:offset+elements])
                offset += elements
                
                # print("offset", offset)
                pass
            assert offset == fused_params.shape[0], (offset, fused_params.shape)
            # if not inside_outside:
            #     torch.nn.init.constant_(self.bias, -bias)
            #     pass
            # else:
            #     torch.nn.init.constant_(self.bias, bias)
            #     pass
            pass
        pass
    
    def forward(self, x):
        # print(self.mlp)
        result = self.mlp(x)
        # print(self.bias)
        # print(list(self.mlp.parameters())[0])
        if self.geometric_init:
            return result + self.bias
        else:
            return result
        # return result
class SDFNetworkNGP(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 bias=0.5,
                 scale=1,
                 ngp_encoding=None,
                 geometric_init=True,
                 weight_norm=False,
                 inside_outside=False,
                 ):
        super(SDFNetworkNGP, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        # if multires > 0:
        #     embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
        #     self.embed_fn_fine = embed_fn
        #     dims[0] = input_ch
        assert d_in == 3, d_in
        
        self.embed_fn_fine = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": ngp_encoding["n_levels"],
                "n_features_per_level": ngp_encoding["n_features_per_level"],
                "log2_hashmap_size": ngp_encoding["log2_hashmap_size"],
                "base_resolution": ngp_encoding["base_resolution"],
                "per_level_scale": ngp_encoding["per_level_scale"],
            },
            # dtype=torch.float32
        )
        self.encoding_length = ngp_encoding["n_features_per_level"] * ngp_encoding["n_levels"]
        dims[0] = self.encoding_length + 3
        
        self.base_enabled_encoding = ngp_encoding["n_features_per_level"] * 4
        self.enabled_encoding = self.encoding_length
        # self.backbone = tcnn.Network(
        #     n_input_dims=32,
        #     n_output_dims=1,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": d_hidden,
        #         "n_hidden_layers": n_layers - 1,
        #     },
        # )

        # Custom initializations too complicated. work on me later.
        assert not weight_norm
        self.scale = scale
        self.mlp = TCNNMLP(
            d_in=dims[0],
            d_out=d_out,
            d_hidden=d_hidden,
            n_layers=n_layers,
            bias=bias,
            scale=scale,
            geometric_init=geometric_init,
            inside_outside=inside_outside
        )
        # self.grad = None
    def set_encoding_level(self, perc_iters):
        # print("perc_iters", perc_iters)
        # print("add", math.floor((self.encoding_length - self.base_enabled_encoding) * perc_iters))
        # if perc_iters > 0.1:
        #     perc_iters -= 0.1
        #     perc_iters = perc_iters / 0.9
        #     self.enabled_encoding = min(self.base_enabled_encoding + math.floor((self.encoding_length) * perc_iters), self.encoding_length)
        # self.enabled_encoding = min(self.base_enabled_encoding + math.floor((self.encoding_length) * perc_iters), self.encoding_length)
        new_encoding = min(self.base_enabled_encoding + math.floor((self.encoding_length) * perc_iters), self.encoding_length)
        if new_encoding != self.enabled_encoding:
            print("new encoding: ", new_encoding)
        print("encoding", new_encoding)
        self.enabled_encoding = new_encoding
        # self.enabled_encoding = self.encoding_length
    def get_encoding(self, inputs):
        # print(inputs.max(), inputs.min())
        inputs = (inputs + 1.1)/2.2
        # print("scale", inputs.max(), inputs.min())
        embed = self.embed_fn_fine(inputs).clone()
        # print("enabled_encoding:", self.enabled_encoding)
        embed[:, self.enabled_encoding:] = 0.0
        # embed = self.embed_fn_fine(inputs)
        return embed
    def forward(self, inputs):
        inputs = inputs * self.scale
        # if self.embed_fn_fine is not None:
        #     inputs = self.embed_fn_fine(inputs)
        # input('before encoding')
        embed = self.get_encoding(inputs)
        # print(inputs.shape)
        # print(embed.shape)
        
        inputs = torch.cat([inputs, embed], dim=1)
        
        x = inputs
        # input('before mlp')
        x = self.mlp(x)
        # print(x.dtype)
        # input('after mlp')
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

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
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
    @torch.enable_grad()
    def sdf_and_gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # print("sdf_and_grad requires_grad", y.requires_grad, gradients.requires_grad)
        return y, gradients.unsqueeze(1)
    
    @torch.enable_grad()
    def hessian(self, x): 
        y,g,h=self.eval_all(x)
        return h
    
    @torch.enable_grad()
    def eval_all(self, x): 
        x.requires_grad_(True)
        y = self(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0] # B x 3
        hess = torch.empty(x.shape[0], 3, 3, dtype=x.dtype, device=x.device)
        for i in range(3):
            # print("i", i)
            hess[:, i, :] = torch.autograd.grad(
                outputs=gradients[:, i:i+1],
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
            )[0]
        # print("all requires_grad", y.requires_grad, gradients.requires_grad, hess.requires_grad)
        return y, gradients, hess

def safe_exp(x):
    return torch.exp(torch.clamp(x, max=10.0))

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
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
                 squeeze_out=True, flip_rgb=False):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        # print("wildlight dims", dims)
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
        self.flip_rgb = flip_rgb
    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None
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

        if self.squeeze_out:
            # x = torch.sigmoid(x)
            x = safe_exp(x)
            pass
        if self.flip_rgb:
            x = torch.flip(x, dims=[-1])
        return x
    
# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
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

    def forward(self, input_pts, input_views):
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
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRFNGP(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
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
        self.embed_fn = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            # dtype=torch.float32
        )
        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view
            
        self.input_ch = 32
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

    def forward(self, input_pts, input_views):
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
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * torch.exp(self.variance * 10.0)
