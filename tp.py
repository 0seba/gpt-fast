# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch import nn

# if os.uname().sysname != "Darwin":
#     from torch.distributed import _functional_collectives as funcol
# else:
#     # Distributed is not supported on MacOS
#     funcol = None
from torch.distributed.tensor.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from model import Attention, FeedForward, Transformer
from quantize import WeightOnlyInt4Linear


def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_local():
    return _get_rank() == 0


def local_break():
    if is_local():
        breakpoint()
    dist.barrier()


def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


# def maybe_init_dist() -> Optional[int]:
#     try:
#         # provided by torchrun
#         rank = _get_rank()
#         world_size = _get_world_size()

#         if world_size < 2:
#             # too few gpus to parallelize, tp is no-op
#             return None
#     except KeyError:
#         # not run via torchrun, no-op
#         return None

#     tp_ranks = list(range(world_size))

#     torch.cuda.set_device(rank)
#     dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
#     tp_group = dist.new_group(ranks=tp_ranks)
#     enable_symm_mem_for_group(tp_group.group_name)
#     torch._inductor.config._micro_pipeline_tp = True

#     return rank


def maybe_init_dist_with_mesh() -> Optional[Tuple[int, DeviceMesh]]:
    try:
        rank = _get_rank()
        world_size = _get_world_size()

        if world_size < 2:
            return None, None
    except KeyError:
        return None, None

    # 1. Initialize the device mesh directly. This replaces dist.init_process_group().
    #    Use world_size to define the mesh shape dynamically.
    tp_mesh = init_device_mesh("cuda", (world_size,))

    # 2. Get the underlying process group from the mesh to use with older APIs.
    tp_group = tp_mesh.get_group()

    # These lines remain the same
    enable_symm_mem_for_group(tp_group.group_name)
    torch._inductor.config._micro_pipeline_tp = True

    # Return the rank and the created mesh for later use
    return rank, tp_mesh


def _apply_tp_linear(
    linear: nn.Linear, style: str, weight_splits: List[int] = []
) -> None:
    rank = _get_rank()
    world_size = _get_world_size()

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0

    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank]

    def shard_qkv(qkv, dim, weight_splits):
        q, k, v = qkv.split(weight_splits, dim=dim)
        q = shard(q, dim)
        k = shard(k, dim)
        v = shard(v, dim)
        return torch.cat((q, k, v), dim=dim)

    # shard
    if weight_splits:
        # attention
        assert len(weight_splits) == 3

        if isinstance(linear, WeightOnlyInt4Linear):
            sharded_weight = shard_qkv(
                linear.weight, shard_dim, [i // 8 for i in weight_splits]
            )
            linear.scales_and_zeros = shard_qkv(
                linear.scales_and_zeros, 1 - shard_dim, weight_splits
            )
        else:
            sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard_qkv(linear.scales, 0, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim)
        if isinstance(linear, WeightOnlyInt4Linear):
            linear.scales_and_zeros = shard(linear.scales_and_zeros, 1 - shard_dim)
            if style == "rowwise":
                assert (
                    linear.scales_and_zeros.shape[0] * 32
                    == sharded_weight.shape[1]
                    * sharded_weight.shape[2]
                    * sharded_weight.shape[3]
                )
                assert linear.scales_and_zeros.shape[1] == sharded_weight.shape[0] * 8
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0)

    # local_break()
    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)

    # shape info should still be synced
    # assert linear.weight.shape == (linear.out_features, linear.in_features)


def _apply_tp_ffn(mlp: FeedForward) -> None:
    assert hasattr(mlp, "w1")
    assert hasattr(mlp, "w3")
    assert hasattr(mlp, "w2")

    _apply_tp_linear(mlp.w1, "colwise")
    _apply_tp_linear(mlp.w3, "colwise")
    _apply_tp_linear(mlp.w2, "rowwise")

    world_size = _get_world_size()
    mlp.register_forward_hook(
        lambda _module, _input, output: funcol.all_reduce(
            output, "sum", list(range(world_size))
        )
    )


def _apply_tp_attn(attn: Attention) -> None:
    assert hasattr(attn, "wqkv")
    assert hasattr(attn, "wo")

    kv_size = attn.n_local_heads * attn.head_dim
    _apply_tp_linear(attn.wqkv, "colwise", [attn.dim, kv_size, kv_size])
    _apply_tp_linear(attn.wo, "rowwise")

    # overwrite
    world_size = _get_world_size()
    attn.n_head = attn.n_head // world_size
    attn.dim = attn.dim // world_size
    attn.head_dim = attn.dim // attn.n_head
    attn.n_local_heads = attn.n_local_heads // world_size

    attn.register_forward_hook(
        lambda _module, _input, output: funcol.all_reduce(
            output[0], "sum", list(range(world_size))
        )
    )


def _apply_tp_Transformer(Transformer: Transformer) -> None:
    # overwrite config before Transformer.setup_cache is called
    world_size = _get_world_size()
    Transformer.config.n_head = Transformer.config.n_head // world_size
    Transformer.config.dim = Transformer.config.dim // world_size
    Transformer.config.n_local_heads = Transformer.config.n_local_heads // world_size


def apply_tp(model: Transformer, tp_mesh: DeviceMesh) -> None:
    _apply_tp_Transformer(model)

    layer_tp_plan = {
        # by default ColwiseParallel input layouts is replicated
        # and RowwiseParallel output layouts is replicated
        "attention.wqkv": ColwiseParallel(),
        # "attention.wk": ColwiseParallel(),
        # "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
    }

    world_size = _get_world_size()

    for block in model.layers:
        wqkv_layer = block.attention.wqkv
        original_weight = wqkv_layer.weight.data

        q_size = block.attention.n_head * block.attention.head_dim
        k_size = block.attention.n_local_heads * block.attention.head_dim
        v_size = block.attention.n_local_heads * block.attention.head_dim
        # Split the weight into their logical Q, K, V parts
        q_w, k_w, v_w = torch.split(original_weight, [q_size, k_size, v_size], dim=0)

        # Reshape each part to isolate the shards for each GPU
        q_w = q_w.view(world_size, q_size // world_size, -1)
        k_w = k_w.view(world_size, k_size // world_size, -1)
        v_w = v_w.view(world_size, v_size // world_size, -1)

        # Interleave the shards and concatenate
        reordered_shards = []
        for i in range(world_size):
            reordered_shards.extend([q_w[i], k_w[i], v_w[i]])

        reordered_weight = torch.cat(reordered_shards, dim=0)

        # Overwrite the original weight with the reordered version
        wqkv_layer.weight.data = reordered_weight

        block.attention.n_head = block.attention.n_head // world_size
        block.attention.dim = block.attention.dim // world_size
        block.attention.head_dim = block.attention.dim // block.attention.n_head
        block.attention.n_local_heads = block.attention.n_local_heads // world_size

        parallelize_module(block, tp_mesh, layer_tp_plan)

        # Apply to MLP
        # _apply_tp_ffn(block.feed_forward)
        # _apply_tp_attn(block.attention)

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "output": ColwiseParallel(
                output_layouts=Replicate(),
            ),
        },
    )
