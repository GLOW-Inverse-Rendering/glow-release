import torch
import collections.abc
def detach_rec(render_out, to_cpu=False):
    for k,v in render_out.items():
        if torch.is_tensor(v):
            render_out[k] = v.detach()
            if to_cpu:
                render_out[k] = render_out[k].cpu()
            
        elif isinstance(v, collections.abc.Mapping):
            render_out[k] = detach_rec(v)


    return render_out
def render_out_to_cuda(render_out):
    for k,v in render_out.items():
        if torch.is_tensor(v):
            render_out[k] = v.cuda()
        elif isinstance(v, int):
            render_out[k] = v
        elif isinstance(v, collections.abc.Mapping):
            render_out[k] = render_out_to_cuda(v)
        else:
            raise RuntimeError(f"Unknown object to detach {v} for key {k}")

    return render_out