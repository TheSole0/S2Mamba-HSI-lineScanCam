# selective_scan.py

import torch

def selective_scan_fn(
    xs,
    dts,
    As,
    Bs,
    Cs,
    Ds,
    delta_bias=None,
    delta_softplus=False,
    context_bias=None
):
    # Dummy version (forward passthrough)
    return xs  # shape (B, C, L)

def selective_scan_ref(
    xs,
    dts,
    As,
    Bs,
    Cs,
    Ds,
    delta_bias=None,
    delta_softplus=False,
    context_bias=None
):
    return xs
