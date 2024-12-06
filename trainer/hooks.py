import re

def parent_module(model, pname):
    comps = pname.split('.')
    parent = model
    for comp in comps[:-1]:
        if hasattr(parent, comp):
            parent = getattr(parent, comp)
        elif comp.isdigit():
            parent = parent[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    assert hasattr(parent, comps[-1])
    return parent


def linear_backward_hook(mod, grad_in, grad_out):
    if not hasattr(mod, "weight"):
        print(f"{mod} has no weight!")
        return

    if hasattr(mod.weight, "__x__"):
        assert len(grad_out) == 1
        # mod.weight.__bgrad__ = grad_out[0].unsqueeze(-1) * mod.__x__[0].unsqueeze(-2)
        mod.weight.__delta__ = grad_out[0].detach()
    else:
        print(f"{mod} has no __x__")


def linear_forward_hook(mod, activations, output):
    assert len(activations) == 1
    mod.weight.__x__ = activations[0].detach()


def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any([re.match(patt, name) for patt in include_patterns])
        valid_type = not include_type or any([isinstance(module, md_cls) for md_cls in include_type])
        if valid_type and valid_name:
            modules[name] = module
    return modules


def hook_model(model, modules):
    handles = []
    for _, m in modules.items():
        if hasattr(m, 'register_full_backward_hook'):
            handles.append(m.register_full_backward_hook(linear_backward_hook))
        else:
            handles.append(m.register_backward_hook(linear_backward_hook))
        handles.append(m.register_forward_hook(linear_forward_hook))

    model.handles = handles