import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _score_mod_signature
from torch._inductor.lowering import make_pointwise, register_lowering
from torch._inductor.virtualized import ops
from functools import partial

@torch.library.custom_op("approx::symlog", mutates_args=())
def _symlog_approx_op(inp: Tensor) -> Tensor:
    """Fallback / Fake kernel for 'approx.symlog'."""
    return torch.sign(inp) * torch.log1p(inp.abs())

@_symlog_approx_op.register_fake
def _(inp: Tensor) -> Tensor:
    """Fallback for fake tensor mode."""
    return torch.sign(inp) * torch.log1p(inp.abs())

def _symlog_approx_lowering(inp):
    """Lower 'approx.symlog' to PTX or another backend."""
    fn = partial(ops.inline_asm_elementwise, asm="symlog.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)

register_lowering(torch.ops.approx.symlog)(_symlog_approx_lowering)

class _SymlogApprox(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.ops.approx.symlog(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        sign_x = torch.sign(x)
        denom = 1.0 + x.abs()
        grad = grad_output * (sign_x / denom)
        return grad

    @staticmethod
    def vmap(info, in_dims, x):
        y = torch.ops.approx.symlog(x)
        return y, 0

_symlog_approx = _SymlogApprox.apply

def symlog_python(x: Tensor) -> Tensor:
    """Pure Python version of symlog."""
    return torch.sign(x) * torch.log1p(x.abs())

def generate_symlog_score_mod(approx: bool = False) -> _score_mod_signature:
    """
    Returns a symlog-based score_mod for Flex Attention.
    If approx=True, uses the custom approx op; else uses python fallback.
    """
    symlog_fn = _symlog_approx if approx else symlog_python
    def symlog_score_mod(score: Tensor, b, h, q_idx, kv_idx) -> Tensor:
        return symlog_fn(score)
    suffix = "approx" if approx else "python"
    symlog_score_mod.__name__ = f"symlog_score_mod_{suffix}"
    return symlog_score_mod

def main(device="cuda"):
    import math
    B, H, SEQ_LEN, D = 1, 1, 12, 16
    query = torch.randn(B, H, SEQ_LEN, D, device=device)
    key = torch.randn(B, H, SEQ_LEN, D, device=device)
    def vanilla_score_mod(score, b, h, q_idx, kv_idx):
        return score
    vanilla_score_mod.__name__ = "vanilla_score_mod"
    def visualize_attention_scores(query: Tensor, key: Tensor, score_mod, device="cuda", name="score_mod_test"):
        B, H, N, D = query.shape
        dot = torch.einsum("bhid,bhjd->bhij", query, key)
        modded = score_mod(dot, 0, 0, None, None)
        attn = torch.softmax(modded, dim=-1)
        attn_2d = attn[0, 0].detach().cpu()
        print(f"==> {name} Attn matrix:\n", attn_2d)
        print(f"    row0 sum: {attn_2d[0].sum().item():.6f}, min: {attn_2d.min():.6f}, max: {attn_2d.max():.6f}\n")
    visualize_attention_scores(query, key, vanilla_score_mod, device, "vanilla")
    symlog_py_mod = generate_symlog_score_mod(approx=False)
    visualize_attention_scores(query, key, symlog_py_mod, device, "symlog_python")
    symlog_approx_mod = generate_symlog_score_mod(approx=True)
    visualize_attention_scores(query, key, symlog_approx_mod, device, "symlog_approx")

if __name__ == "__main__":
    main("cuda" if torch.cuda.is_available() else "cpu")
