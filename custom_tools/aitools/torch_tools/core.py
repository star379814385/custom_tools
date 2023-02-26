try:
    import torch
except:
    raise ImportError("please install torch.")

__all__ = [
    "set_nn_module",
]

def set_nn_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)