# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/oh/coh73potub4z4drg6ujjwhabxtv5fmv2lmza36fttoj5x4ux6tin.py
# Topologically Sorted Source Nodes: [abs_2, radius_1, ly, uy], Original ATen: [aten.abs, aten.sum, aten.sub, aten.add]
# Source node to ATen node mapping:
#   abs_2 => abs_2
#   ly => sub_3
#   radius_1 => sum_2
#   uy => add_9
# Graph fragment:
#   %arg5_1 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg3_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %sum_2 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sum_2]
#   %abs_2 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg5_1,), kwargs = {})
#   %sum_2 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_2, [1]), kwargs = {})
#   %sub_3 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   %add_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   return %sum_2,%sub_3,%add_9
triton_red_fused_abs_add_sub_sum_0 = async_compile.triton('triton_red_fused_abs_add_sub_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 2},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'out_ptr1': '*fp64', 'out_ptr2': '*fp64', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_add_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1835008, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_abs_add_sub_sum_0(in_ptr0, in_ptr1, out_ptr1, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float64)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 1024*ks0*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl_math.abs(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp6 = tmp5 - tmp3
    tmp7 = tmp5 + tmp3
    tl.store(out_ptr1 + (x3), tmp6, None)
    tl.store(out_ptr2 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/yt/cyt6jc2gxiajecdmb5bpgssotr5vjga5ljbx46hptgpbbwpbtsqo.py
# Topologically Sorted Source Nodes: [abs_1, radius, lx, ux, s_lb, sub_2, mul, s_ub, sub_3, mul_1, lam, mul_6, add_2, add_3, mul_2, sub_4, mu1, sig_c, copy_, sub_5, sub_6, mul_4, sub_7, mu2], Original ATen: [aten.abs, aten.sum, aten.sub, aten.add, aten.sigmoid, aten.rsub, aten.mul, aten.minimum, aten.slice, aten.diagonal, aten.copy]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   add_2 => add_10
#   add_3 => add_11
#   copy_ => copy_3, diagonal_1, slice_17
#   lam => minimum
#   lx => sub_1
#   mu1 => mul_9
#   mu2 => mul_11
#   mul => mul_6
#   mul_1 => mul_7
#   mul_2 => mul_8
#   mul_4 => mul_10
#   mul_6 => mul_12
#   radius => sum_1
#   s_lb => sigmoid
#   s_ub => sigmoid_1
#   sig_c => add_12
#   sub_2 => sub_4
#   sub_3 => sub_5
#   sub_4 => sub_6
#   sub_5 => sub_7
#   sub_6 => sub_8
#   sub_7 => sub_9
#   ux => add_4
# Graph fragment:
#   %arg1_1 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %sum_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %sub_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sub_1]
#   %add_4 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=add_4]
#   %abs_1 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg1_1,), kwargs = {})
#   %sum_1 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [1]), kwargs = {})
#   %sub_1 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %add_4 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %sigmoid : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%sub_1,), kwargs = {})
#   %sub_4 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_6 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub_4), kwargs = {})
#   %sigmoid_1 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_4,), kwargs = {})
#   %sub_5 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_1), kwargs = {})
#   %mul_7 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %sub_5), kwargs = {})
#   %minimum : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.minimum.default](args = (%mul_6, %mul_7), kwargs = {})
#   %mul_12 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%minimum, %arg2_1), kwargs = {})
#   %add_10 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid_1, %sigmoid), kwargs = {})
#   %add_11 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %sub_1), kwargs = {})
#   %mul_8 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%minimum, %add_11), kwargs = {})
#   %sub_6 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_10, %mul_8), kwargs = {})
#   %mul_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, 0.5), kwargs = {})
#   %add_12 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_9), kwargs = {})
#   %slice_17 : Tensor "f64[32, 1024, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_2, 1, %arg4_1, %add_88), kwargs = {})
#   %diagonal_1 : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.diagonal.default](args = (%slice_17, 0, 1, 2), kwargs = {})
#   %sub_7 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sigmoid_1, %sigmoid), kwargs = {})
#   %sub_8 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %sub_1), kwargs = {})
#   %mul_10 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%minimum, %sub_8), kwargs = {})
#   %sub_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %mul_10), kwargs = {})
#   %mul_11 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, 0.5), kwargs = {})
#   %copy_3 : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%diagonal_1, %mul_11), kwargs = {})
#   return %sum_1,%sub_1,%add_4,%add_12,%copy_3
triton_red_fused_abs_add_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_sum_1 = async_compile.triton('triton_red_fused_abs_add_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 2},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'out_ptr1': '*fp64', 'out_ptr2': '*fp64', 'out_ptr3': '*fp64', 'out_ptr4': '*fp64', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_add_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 4, 'num_reduction': 1, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2883584, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_abs_add_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_sum_1(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float64)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 1024*ks0*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl_math.abs(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp6 = tmp5 - tmp3
    tmp7 = tmp5 + tmp3
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = -tmp6
    tmp10 = libdevice.exp(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = (tmp8 / tmp11)
    tmp13 = tl.full([1, 1], 1.0, tl.float64)
    tmp14 = tmp13 - tmp12
    tmp15 = tmp12 * tmp14
    tmp16 = -tmp7
    tmp17 = libdevice.exp(tmp16)
    tmp18 = tmp8 + tmp17
    tmp19 = (tmp8 / tmp18)
    tmp20 = tmp13 - tmp19
    tmp21 = tmp19 * tmp20
    tmp22 = triton_helpers.minimum(tmp15, tmp21)
    tmp23 = tmp22 * tmp5
    tmp24 = tmp19 + tmp12
    tmp25 = tmp7 + tmp6
    tmp26 = tmp22 * tmp25
    tmp27 = tmp24 - tmp26
    tmp28 = tl.full([1, 1], 0.5, tl.float64)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp23 + tmp29
    tmp31 = tmp19 - tmp12
    tmp32 = tmp7 - tmp6
    tmp33 = tmp22 * tmp32
    tmp34 = tmp31 - tmp33
    tmp35 = tmp34 * tmp28
    tl.store(out_ptr1 + (x3), tmp6, None)
    tl.store(out_ptr2 + (x3), tmp7, None)
    tl.store(out_ptr3 + (x3), tmp30, None)
    tl.store(out_ptr4 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/xm/cxmm56n3lx3ne32z3rf5ezydrxnx5ntsrk4zyhwdbqggmpjpobsj.py
# Topologically Sorted Source Nodes: [lx, lx_f, unsqueeze, t, ux, ux_f, sub_8, step, unsqueeze_1, mul_7, out, setitem, unsqueeze_2, getitem_3, sub_9, unsqueeze_3, mul_8, sub_10, repeat_interleave, grid, ly, ly_f, unsqueeze_4, t_1, uy, uy_f, sub_11, step_1, unsqueeze_5, mul_9, out_1, setitem_1, unsqueeze_6, getitem_4, sub_12, unsqueeze_7, mul_10, sub_13, repeat], Original ATen: [aten.sub, aten.view, aten.unsqueeze, aten.arange, aten.add, aten.div, aten.mul, aten.slice, aten.rsub, aten.copy, aten.expand, aten.clone, aten.stack, aten.repeat]
# Source node to ATen node mapping:
#   getitem_3 => slice_1
#   getitem_4 => slice_4
#   grid => cat, unsqueeze_11, unsqueeze_12
#   lx => sub_1
#   lx_f => view
#   ly => sub_3
#   ly_f => view_2
#   mul_10 => mul_18
#   mul_7 => mul_14
#   mul_8 => mul_15
#   mul_9 => mul_17
#   out => add_14
#   out_1 => add_16
#   repeat => repeat
#   repeat_interleave => clone, expand_1, unsqueeze_10, view_4
#   setitem => copy, slice_2
#   setitem_1 => copy_1, slice_5
#   step => div
#   step_1 => div_1
#   sub_10 => sub_12
#   sub_11 => sub_13
#   sub_12 => sub_14
#   sub_13 => sub_15
#   sub_8 => sub_10
#   sub_9 => sub_11
#   t => add_13, convert_element_type, iota, mul_13
#   t_1 => add_15, convert_element_type_1, iota_1, mul_16
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
#   unsqueeze_4 => unsqueeze_4
#   unsqueeze_5 => unsqueeze_5
#   unsqueeze_6 => unsqueeze_6
#   unsqueeze_7 => unsqueeze_7
#   ux => add_4
#   ux_f => view_1
#   uy => add_9
#   uy_f => view_3
# Graph fragment:
#   %add_4 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=add_4]
#   %sub_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sub_1]
#   %add_9 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=add_9]
#   %sub_3 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sub_3]
#   %sub_1 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %view : Tensor "f64[32768][1]cuda:0"[num_users=11] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_1, [-1]), kwargs = {})
#   %unsqueeze : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view, 1), kwargs = {})
#   %iota : Tensor "i64[5][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (5,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_13 : Tensor "i64[5][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_13 : Tensor "i64[5][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, 0), kwargs = {})
#   %convert_element_type : Tensor "f64[5][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.float64), kwargs = {})
#   %add_4 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %view_1 : Tensor "f64[32768][1]cuda:0"[num_users=11] = call_function[target=torch.ops.aten.reshape.default](args = (%add_4, [-1]), kwargs = {})
#   %sub_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %view), kwargs = {})
#   %div : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_10, 4), kwargs = {})
#   %unsqueeze_1 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div, 1), kwargs = {})
#   %mul_14 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %unsqueeze_1), kwargs = {})
#   %add_14 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, %mul_14), kwargs = {})
#   %slice_2 : Tensor "f64[32768, 3][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_14, 1, 2, 9223372036854775807), kwargs = {})
#   %unsqueeze_2 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_1, 1), kwargs = {})
#   %slice_1 : Tensor "f64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type, 0, 2, 9223372036854775807), kwargs = {})
#   %sub_11 : Tensor "f64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (4, %slice_1), kwargs = {})
#   %unsqueeze_3 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div, 1), kwargs = {})
#   %mul_15 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_3), kwargs = {})
#   %sub_12 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %mul_15), kwargs = {})
#   %copy : Tensor "f64[32768, 3][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_2, %sub_12), kwargs = {})
#   %slice_scatter_default : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%add_14, %copy, 1, 2, 9223372036854775807), kwargs = {})
#   %unsqueeze_10 : Tensor "f64[32768, 5, 1][5, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_scatter_default, 2), kwargs = {})
#   %expand_1 : Tensor "f64[32768, 5, 5][5, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_10, [32768, 5, 5]), kwargs = {})
#   %clone : Tensor "f64[32768, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_4 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [32768, 25]), kwargs = {})
#   %unsqueeze_11 : Tensor "f64[32768, 25, 1][25, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_4, 2), kwargs = {})
#   %sub_3 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   %view_2 : Tensor "f64[32768][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_3, [-1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 1), kwargs = {})
#   %iota_1 : Tensor "i64[5][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (5,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_16 : Tensor "i64[5][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_15 : Tensor "i64[5][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, 0), kwargs = {})
#   %convert_element_type_1 : Tensor "f64[5][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.float64), kwargs = {})
#   %add_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   %view_3 : Tensor "f64[32768][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%add_9, [-1]), kwargs = {})
#   %sub_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %view_2), kwargs = {})
#   %div_1 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_13, 4), kwargs = {})
#   %unsqueeze_5 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div_1, 1), kwargs = {})
#   %mul_17 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %unsqueeze_5), kwargs = {})
#   %add_16 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_4, %mul_17), kwargs = {})
#   %slice_5 : Tensor "f64[32768, 3][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_16, 1, 2, 9223372036854775807), kwargs = {})
#   %unsqueeze_6 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 1), kwargs = {})
#   %slice_4 : Tensor "f64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1, 0, 2, 9223372036854775807), kwargs = {})
#   %sub_14 : Tensor "f64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (4, %slice_4), kwargs = {})
#   %unsqueeze_7 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div_1, 1), kwargs = {})
#   %mul_18 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_7), kwargs = {})
#   %sub_15 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_6, %mul_18), kwargs = {})
#   %copy_1 : Tensor "f64[32768, 3][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %sub_15), kwargs = {})
#   %slice_scatter_default_1 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%add_16, %copy_1, 1, 2, 9223372036854775807), kwargs = {})
#   %repeat : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_scatter_default_1, [1, 5]), kwargs = {})
#   %unsqueeze_12 : Tensor "f64[32768, 25, 1][25, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%repeat, 2), kwargs = {})
#   %cat : Tensor "f64[32768, 25, 2][50, 2, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_11, %unsqueeze_12], 2), kwargs = {})
#   return %cat
triton_poi_fused_add_arange_clone_copy_div_expand_mul_repeat_rsub_slice_stack_sub_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_arange_clone_copy_div_expand_mul_repeat_rsub_slice_stack_sub_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'out_ptr0': '*fp64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_clone_copy_div_expand_mul_repeat_rsub_slice_stack_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1048576, 'x': 26214400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_clone_copy_div_expand_mul_repeat_rsub_slice_stack_sub_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 50
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 2)
    x2 = xindex // 2
    y0 = yindex
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(x2 // 5, [YBLOCK, XBLOCK])
    tmp6 = tl.full([1, 1], 2, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([1, 1], 4.0, tl.float64)
    tmp11 = tl.broadcast_to(x2 // 5, [YBLOCK, XBLOCK])
    tmp12 = tmp11.to(tl.float64)
    tmp13 = tmp10 - tmp12
    tmp14 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 - tmp14
    tmp16 = tl.full([1, 1], 0.25, tl.float64)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp13 * tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp22 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp23 - tmp22
    tmp25 = tl.full([1, 1], 0.25, tl.float64)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp5.to(tl.float64)
    tmp28 = tmp27 * tmp26
    tmp29 = tmp22 + tmp28
    tmp30 = tl.where(tmp7, tmp21, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp4, tmp30, tmp31)
    tmp33 = tmp0 >= tmp3
    tmp34 = tl.full([1, 1], 2, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tl.broadcast_to(((x3 // 2) % 5), [YBLOCK, XBLOCK])
    tmp37 = tl.full([1, 1], 2, tl.int64)
    tmp38 = tmp36 >= tmp37
    tmp39 = tmp38 & tmp33
    tmp40 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full([1, 1], 4.0, tl.float64)
    tmp42 = tl.broadcast_to(((x3 // 2) % 5), [YBLOCK, XBLOCK])
    tmp43 = tmp42.to(tl.float64)
    tmp44 = tmp41 - tmp43
    tmp45 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp40 - tmp45
    tmp47 = tl.full([1, 1], 0.25, tl.float64)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp44 * tmp48
    tmp50 = tmp40 - tmp49
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp39, tmp50, tmp51)
    tmp53 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp54 - tmp53
    tmp56 = tl.full([1, 1], 0.25, tl.float64)
    tmp57 = tmp55 * tmp56
    tmp58 = tmp36.to(tl.float64)
    tmp59 = tmp58 * tmp57
    tmp60 = tmp53 + tmp59
    tmp61 = tl.where(tmp38, tmp52, tmp60)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp33, tmp61, tmp62)
    tmp64 = tl.where(tmp4, tmp32, tmp63)
    tl.store(out_ptr0 + (x3 + 50*y0), tmp64, xmask)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/cz/cczvqkx2raw2sygmpg5xpzlxm42q7h3oukhzuq2exsfd5dz3dywn.py
# Topologically Sorted Source Nodes: [gx, sigmoid_2, sub_14, sigmoid_3, gy, mul_11, targets, z3, getitem_25, sub_53, px, getitem_26, sub_55, py], Original ATen: [aten.select, aten.sigmoid, aten.rsub, aten.mul, aten.view, aten.slice, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   getitem_25 => slice_11
#   getitem_26 => slice_12
#   gx => select
#   gy => select_1
#   mul_11 => mul_19
#   px => mean_3
#   py => mean_5
#   sigmoid_2 => sigmoid_2
#   sigmoid_3 => sigmoid_3
#   sub_14 => sub_16
#   sub_53 => sub_55
#   sub_55 => sub_57
#   targets => mul_20
#   z3 => view_5
# Graph fragment:
#   %cat : Tensor "f64[32768, 25, 2][50, 2, 1]cuda:0" = PlaceHolder[target=cat]
#   %select : Tensor "f64[32768, 25][50, 2]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%cat, 2, 0), kwargs = {})
#   %sigmoid_2 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select,), kwargs = {})
#   %sub_16 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid_2), kwargs = {})
#   %sigmoid_3 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select,), kwargs = {})
#   %select_1 : Tensor "f64[32768, 25][50, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat, 2, 1), kwargs = {})
#   %mul_19 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_3, %select_1), kwargs = {})
#   %mul_20 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %mul_19), kwargs = {})
#   %view_5 : Tensor "f64[32768, 5, 5][25, 5, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_20, [32768, 5, 5]), kwargs = {})
#   %slice_11 : Tensor "f64[32768, 1, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_5, 1, 0, 1), kwargs = {})
#   %sub_55 : Tensor "f64[32768, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %slice_11), kwargs = {})
#   %mean_3 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_55, [2]), kwargs = {})
#   %slice_12 : Tensor "f64[32768, 5, 1][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_5, 2, 0, 1), kwargs = {})
#   %sub_57 : Tensor "f64[32768, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %slice_12), kwargs = {})
#   %mean_5 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_57, [1]), kwargs = {})
#   return %mean_3,%mean_5
triton_poi_fused_mean_mul_rsub_select_sigmoid_slice_sub_view_3 = async_compile.triton('triton_poi_fused_mean_mul_rsub_select_sigmoid_slice_sub_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'out_ptr1': '*fp64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_rsub_select_sigmoid_slice_sub_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 38, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 5242880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_rsub_select_sigmoid_slice_sub_view_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 5
    x0 = (xindex % 5)
    tmp0 = tl.load(in_ptr0 + (10*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 10*x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (50*x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (1 + 50*x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (2 + 10*x2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (3 + 10*x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (2 + 50*x1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (3 + 50*x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (4 + 10*x2), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (5 + 10*x2), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (4 + 50*x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr0 + (5 + 50*x1), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (6 + 10*x2), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr0 + (7 + 10*x2), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr0 + (6 + 50*x1), None, eviction_policy='evict_last')
    tmp76 = tl.load(in_ptr0 + (7 + 50*x1), None, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (8 + 10*x2), None, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr0 + (9 + 10*x2), None, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr0 + (8 + 50*x1), None, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr0 + (9 + 50*x1), None, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr0 + (2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr0 + (1 + 2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr0 + (10 + 2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp119 = tl.load(in_ptr0 + (11 + 2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr0 + (10 + 50*x1), None, eviction_policy='evict_last')
    tmp128 = tl.load(in_ptr0 + (11 + 50*x1), None, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr0 + (20 + 2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr0 + (21 + 2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp142 = tl.load(in_ptr0 + (20 + 50*x1), None, eviction_policy='evict_last')
    tmp148 = tl.load(in_ptr0 + (21 + 50*x1), None, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (30 + 2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr0 + (31 + 2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr0 + (30 + 50*x1), None, eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr0 + (31 + 50*x1), None, eviction_policy='evict_last')
    tmp173 = tl.load(in_ptr0 + (40 + 2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp179 = tl.load(in_ptr0 + (41 + 2*x0 + 50*x1), None, eviction_policy='evict_last')
    tmp182 = tl.load(in_ptr0 + (40 + 50*x1), None, eviction_policy='evict_last')
    tmp188 = tl.load(in_ptr0 + (41 + 50*x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = -tmp0
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tmp1 + tmp3
    tmp5 = (tmp1 / tmp4)
    tmp6 = tl.full([1], 1.0, tl.float64)
    tmp7 = tmp6 - tmp5
    tmp9 = tmp5 * tmp8
    tmp10 = tmp7 * tmp9
    tmp12 = -tmp11
    tmp13 = libdevice.exp(tmp12)
    tmp14 = tmp1 + tmp13
    tmp15 = (tmp1 / tmp14)
    tmp16 = tmp6 - tmp15
    tmp18 = tmp15 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp10 - tmp19
    tmp22 = -tmp21
    tmp23 = libdevice.exp(tmp22)
    tmp24 = tmp1 + tmp23
    tmp25 = (tmp1 / tmp24)
    tmp26 = tmp6 - tmp25
    tmp28 = tmp25 * tmp27
    tmp29 = tmp26 * tmp28
    tmp31 = -tmp30
    tmp32 = libdevice.exp(tmp31)
    tmp33 = tmp1 + tmp32
    tmp34 = (tmp1 / tmp33)
    tmp35 = tmp6 - tmp34
    tmp37 = tmp34 * tmp36
    tmp38 = tmp35 * tmp37
    tmp39 = tmp29 - tmp38
    tmp40 = tmp20 + tmp39
    tmp42 = -tmp41
    tmp43 = libdevice.exp(tmp42)
    tmp44 = tmp1 + tmp43
    tmp45 = (tmp1 / tmp44)
    tmp46 = tmp6 - tmp45
    tmp48 = tmp45 * tmp47
    tmp49 = tmp46 * tmp48
    tmp51 = -tmp50
    tmp52 = libdevice.exp(tmp51)
    tmp53 = tmp1 + tmp52
    tmp54 = (tmp1 / tmp53)
    tmp55 = tmp6 - tmp54
    tmp57 = tmp54 * tmp56
    tmp58 = tmp55 * tmp57
    tmp59 = tmp49 - tmp58
    tmp60 = tmp40 + tmp59
    tmp62 = -tmp61
    tmp63 = libdevice.exp(tmp62)
    tmp64 = tmp1 + tmp63
    tmp65 = (tmp1 / tmp64)
    tmp66 = tmp6 - tmp65
    tmp68 = tmp65 * tmp67
    tmp69 = tmp66 * tmp68
    tmp71 = -tmp70
    tmp72 = libdevice.exp(tmp71)
    tmp73 = tmp1 + tmp72
    tmp74 = (tmp1 / tmp73)
    tmp75 = tmp6 - tmp74
    tmp77 = tmp74 * tmp76
    tmp78 = tmp75 * tmp77
    tmp79 = tmp69 - tmp78
    tmp80 = tmp60 + tmp79
    tmp82 = -tmp81
    tmp83 = libdevice.exp(tmp82)
    tmp84 = tmp1 + tmp83
    tmp85 = (tmp1 / tmp84)
    tmp86 = tmp6 - tmp85
    tmp88 = tmp85 * tmp87
    tmp89 = tmp86 * tmp88
    tmp91 = -tmp90
    tmp92 = libdevice.exp(tmp91)
    tmp93 = tmp1 + tmp92
    tmp94 = (tmp1 / tmp93)
    tmp95 = tmp6 - tmp94
    tmp97 = tmp94 * tmp96
    tmp98 = tmp95 * tmp97
    tmp99 = tmp89 - tmp98
    tmp100 = tmp80 + tmp99
    tmp101 = tl.full([1], 5.0, tl.float64)
    tmp102 = (tmp100 / tmp101)
    tmp104 = -tmp103
    tmp105 = libdevice.exp(tmp104)
    tmp106 = tmp1 + tmp105
    tmp107 = (tmp1 / tmp106)
    tmp108 = tmp6 - tmp107
    tmp110 = tmp107 * tmp109
    tmp111 = tmp108 * tmp110
    tmp112 = tmp111 - tmp19
    tmp114 = -tmp113
    tmp115 = libdevice.exp(tmp114)
    tmp116 = tmp1 + tmp115
    tmp117 = (tmp1 / tmp116)
    tmp118 = tmp6 - tmp117
    tmp120 = tmp117 * tmp119
    tmp121 = tmp118 * tmp120
    tmp123 = -tmp122
    tmp124 = libdevice.exp(tmp123)
    tmp125 = tmp1 + tmp124
    tmp126 = (tmp1 / tmp125)
    tmp127 = tmp6 - tmp126
    tmp129 = tmp126 * tmp128
    tmp130 = tmp127 * tmp129
    tmp131 = tmp121 - tmp130
    tmp132 = tmp112 + tmp131
    tmp134 = -tmp133
    tmp135 = libdevice.exp(tmp134)
    tmp136 = tmp1 + tmp135
    tmp137 = (tmp1 / tmp136)
    tmp138 = tmp6 - tmp137
    tmp140 = tmp137 * tmp139
    tmp141 = tmp138 * tmp140
    tmp143 = -tmp142
    tmp144 = libdevice.exp(tmp143)
    tmp145 = tmp1 + tmp144
    tmp146 = (tmp1 / tmp145)
    tmp147 = tmp6 - tmp146
    tmp149 = tmp146 * tmp148
    tmp150 = tmp147 * tmp149
    tmp151 = tmp141 - tmp150
    tmp152 = tmp132 + tmp151
    tmp154 = -tmp153
    tmp155 = libdevice.exp(tmp154)
    tmp156 = tmp1 + tmp155
    tmp157 = (tmp1 / tmp156)
    tmp158 = tmp6 - tmp157
    tmp160 = tmp157 * tmp159
    tmp161 = tmp158 * tmp160
    tmp163 = -tmp162
    tmp164 = libdevice.exp(tmp163)
    tmp165 = tmp1 + tmp164
    tmp166 = (tmp1 / tmp165)
    tmp167 = tmp6 - tmp166
    tmp169 = tmp166 * tmp168
    tmp170 = tmp167 * tmp169
    tmp171 = tmp161 - tmp170
    tmp172 = tmp152 + tmp171
    tmp174 = -tmp173
    tmp175 = libdevice.exp(tmp174)
    tmp176 = tmp1 + tmp175
    tmp177 = (tmp1 / tmp176)
    tmp178 = tmp6 - tmp177
    tmp180 = tmp177 * tmp179
    tmp181 = tmp178 * tmp180
    tmp183 = -tmp182
    tmp184 = libdevice.exp(tmp183)
    tmp185 = tmp1 + tmp184
    tmp186 = (tmp1 / tmp185)
    tmp187 = tmp6 - tmp186
    tmp189 = tmp186 * tmp188
    tmp190 = tmp187 * tmp189
    tmp191 = tmp181 - tmp190
    tmp192 = tmp172 + tmp191
    tmp193 = (tmp192 / tmp101)
    tl.store(out_ptr0 + (x2), tmp102, None)
    tl.store(out_ptr1 + (x2), tmp193, None)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/uo/cuohpa5nnb3jflp6g7iovkk7je7sjk5yamygohtuvqfoqco266k7.py
# Topologically Sorted Source Nodes: [xs, getitem_9, r, rm, dx, abs_3, m, frexp, ys, getitem_12, r_1, rm_1, dy, abs_4, m_1, frexp_1, gx, sigmoid_2, sub_14, sigmoid_3, gy, mul_11, targets, mean_4, px_1, abs_7, m_3, frexp_2, mean_6, py_1, abs_8, m_4, frexp_3, zbar, zero_1, ones_like, ax, getitem_19, dxn, ones_like_12, sx, getitem_31, truediv_24, mul_69, sum_7, getitem_21, getitem_22, eq, mul_15, sum_5, eq_1, xdeg, one, getitem_10, getitem_11, xbar, abs_5, getitem_13, getitem_14, ybar, abs_6, maximum, maximum_1, mul_13, nxx, truediv_4, sqrt, mul_17, ones_like_1, ay, getitem_20, dyn, mul_14, nyy, truediv_5, sqrt_1, mul_18, maximum_2, m_2, truediv_6, c2, truediv_7, a2, add_9, truediv_8, b2, add_10, truediv_9, p2, add_11, truediv_10, q2, e1, mul_26, mul_27, sub_21, mul_28, add_13, mul_19, add_14, mul_20, add_15, mul_21, e2, df, gt, sub_19, mul_24, add_17, mul_25, mul_22, e3, f, le, ones_like_3, where, truediv_11, zero, where_1, l1, mul_31, mul_32, sub_25, mul_33, df_1, gt_1, sub_23, mul_29, add_19, mul_30, f_1, le_1, ones_like_4, where_2, truediv_12, where_3, l1_1, mul_36, mul_37, sub_29, mul_38, df_2, gt_2, sub_27, mul_34, add_21, mul_35, f_2, le_2, ones_like_5, where_4, truediv_13, where_5, l1_2, mul_41, mul_42, sub_33, mul_43, df_3, gt_3, sub_31, mul_39, add_23, mul_40, f_3, le_3, ones_like_6, where_6, truediv_14, where_7, l1_3, mul_46, mul_47, sub_37, mul_48, df_4, gt_4, sub_35, mul_44, add_25, mul_45, f_4, le_4, ones_like_7, where_8, truediv_15, where_9, l1_4, mul_51, mul_52, sub_41, mul_53, df_5, gt_5, sub_39, mul_49, add_27, mul_50, f_5, le_5, ones_like_8, where_10, truediv_16, where_11, l1_5, mul_56, mul_57, sub_45, mul_58, df_6, gt_6, sub_43, mul_54, add_29, mul_55, f_6, le_6, ones_like_9, where_12, truediv_17, where_13, l1_6, mul_61, mul_62, sub_49, mul_63, df_7, gt_7, sub_47, mul_59, add_31, mul_60, f_7, le_7, ones_like_10, where_14, truediv_18, where_15, l1_7, prod, sub_51, truediv_20, total, mul_64, mul_65, sub_52, disc, sqrt_2, add_33, l2, mul_67, drop2, gt_8, le_8, ones_like_11, where_16, truediv_21, l3, mul_68, drop3, add_34, tx, add_35, ty, pick_x, and_, cx, or__4, ones_like_14, where_18, truediv_25, mul_70, ones_like_15, where_19, truediv_26, a0, mul_73, sub_57, ones_like_13, sy, getitem_32, truediv_27, mul_71, sum_8, getitem_23, getitem_24, eq_2, mul_16, sum_6, eq_3, ydeg, invert, and__1, cy, or__5, ones_like_16, where_21, truediv_28, mul_72, ones_like_17, where_22, truediv_29, b0, mul_74, k, ones_like_18, zero_2, u, abs_9, v, abs_10, maximum_4, m_5, km, inv, mul_75, un, mul_76, add_36, vn, mul_77, denom], Original ATen: [aten.slice, aten.select, aten.sub, aten.mean, aten.abs, aten.amax, aten.frexp, aten.sigmoid, aten.rsub, aten.mul, aten.zeros_like, aten.ones_like, aten.ldexp, aten.unsqueeze, aten.div, aten.sum, aten.eq, aten.bitwise_or, aten.add, aten.maximum, aten.sqrt, aten.pow, aten.gt, aten.le, aten.where, aten.clamp_min, aten.bitwise_and, aten.bitwise_not, aten.reciprocal]
# Source node to ATen node mapping:
#   a0 => where_20
#   a2 => pow_2
#   abs_10 => abs_10
#   abs_3 => abs_3
#   abs_4 => abs_4
#   abs_5 => abs_5
#   abs_6 => abs_6
#   abs_7 => abs_7
#   abs_8 => abs_8
#   abs_9 => abs_9
#   add_10 => add_20
#   add_11 => add_21
#   add_13 => add_23
#   add_14 => add_24
#   add_15 => add_25
#   add_17 => add_27
#   add_19 => add_29
#   add_21 => add_31
#   add_23 => add_33
#   add_25 => add_35
#   add_27 => add_37
#   add_29 => add_39
#   add_31 => add_41
#   add_33 => add_43
#   add_34 => add_44
#   add_35 => add_45
#   add_36 => add_46
#   add_9 => add_19
#   and_ => bitwise_and
#   and__1 => bitwise_and_1
#   ax => ldexp
#   ay => ldexp_1
#   b0 => where_23
#   b2 => pow_3
#   c2 => pow_1
#   cx => bitwise_or_2
#   cy => bitwise_or_3
#   denom => add_47
#   df => add_28
#   df_1 => add_30
#   df_2 => add_32
#   df_3 => add_34
#   df_4 => add_36
#   df_5 => add_38
#   df_6 => add_40
#   df_7 => add_42
#   disc => clamp_min_1
#   drop2 => le_9
#   drop3 => le_10
#   dx => sub_18
#   dxn => div_2
#   dy => sub_20
#   dyn => div_3
#   e1 => add_22
#   e2 => add_26
#   e3 => mul_31
#   eq => eq_2
#   eq_1 => eq_3
#   eq_2 => eq_4
#   eq_3 => eq_5
#   f => sub_22
#   f_1 => sub_26
#   f_2 => sub_30
#   f_3 => sub_34
#   f_4 => sub_38
#   f_5 => sub_42
#   f_6 => sub_46
#   f_7 => sub_50
#   frexp => frexp
#   frexp_1 => frexp_1
#   frexp_2 => frexp_2
#   frexp_3 => frexp_3
#   getitem_10 => select_4
#   getitem_11 => select_5
#   getitem_12 => slice_10
#   getitem_13 => select_6
#   getitem_14 => select_7
#   getitem_19 => unsqueeze_13
#   getitem_20 => unsqueeze_14
#   getitem_21 => select_8
#   getitem_22 => select_9
#   getitem_23 => select_10
#   getitem_24 => select_11
#   getitem_31 => unsqueeze_15
#   getitem_32 => unsqueeze_16
#   getitem_9 => slice_9
#   gt => gt
#   gt_1 => gt_1
#   gt_2 => gt_2
#   gt_3 => gt_3
#   gt_4 => gt_4
#   gt_5 => gt_5
#   gt_6 => gt_6
#   gt_7 => gt_7
#   gt_8 => gt_8
#   gx => select
#   gy => select_1
#   inv => mul_83, reciprocal
#   invert => bitwise_not
#   k => sub_60
#   km => div_32
#   l1 => sub_24
#   l1_1 => sub_28
#   l1_2 => sub_32
#   l1_3 => sub_36
#   l1_4 => sub_40
#   l1_5 => sub_44
#   l1_6 => sub_48
#   l1_7 => sub_52
#   l2 => mul_74
#   l3 => where_17
#   le => le
#   le_1 => le_1
#   le_2 => le_2
#   le_3 => le_3
#   le_4 => le_4
#   le_5 => le_5
#   le_6 => le_6
#   le_7 => le_7
#   le_8 => le_8
#   m => amax
#   m_1 => amax_1
#   m_2 => maximum_3
#   m_3 => amax_2
#   m_4 => amax_3
#   m_5 => maximum_5
#   maximum => maximum
#   maximum_1 => maximum_1
#   maximum_2 => maximum_2
#   maximum_4 => maximum_4
#   mean_4 => mean_4
#   mean_6 => mean_6
#   mul_11 => mul_19
#   mul_13 => mul_21
#   mul_14 => mul_22
#   mul_15 => mul_23
#   mul_16 => mul_24
#   mul_17 => mul_25
#   mul_18 => mul_26
#   mul_19 => mul_27
#   mul_20 => mul_28
#   mul_21 => mul_29
#   mul_22 => mul_30
#   mul_24 => mul_32
#   mul_25 => mul_33
#   mul_26 => mul_34
#   mul_27 => mul_35
#   mul_28 => mul_36
#   mul_29 => mul_37
#   mul_30 => mul_38
#   mul_31 => mul_39
#   mul_32 => mul_40
#   mul_33 => mul_41
#   mul_34 => mul_42
#   mul_35 => mul_43
#   mul_36 => mul_44
#   mul_37 => mul_45
#   mul_38 => mul_46
#   mul_39 => mul_47
#   mul_40 => mul_48
#   mul_41 => mul_49
#   mul_42 => mul_50
#   mul_43 => mul_51
#   mul_44 => mul_52
#   mul_45 => mul_53
#   mul_46 => mul_54
#   mul_47 => mul_55
#   mul_48 => mul_56
#   mul_49 => mul_57
#   mul_50 => mul_58
#   mul_51 => mul_59
#   mul_52 => mul_60
#   mul_53 => mul_61
#   mul_54 => mul_62
#   mul_55 => mul_63
#   mul_56 => mul_64
#   mul_57 => mul_65
#   mul_58 => mul_66
#   mul_59 => mul_67
#   mul_60 => mul_68
#   mul_61 => mul_69
#   mul_62 => mul_70
#   mul_63 => mul_71
#   mul_64 => mul_72
#   mul_65 => mul_73
#   mul_67 => mul_75
#   mul_68 => mul_76
#   mul_69 => mul_77
#   mul_70 => mul_78
#   mul_71 => mul_79
#   mul_72 => mul_80
#   mul_73 => mul_81
#   mul_74 => mul_82
#   mul_75 => mul_84
#   mul_76 => mul_85
#   mul_77 => mul_86
#   nxx => sum_3
#   nyy => sum_4
#   one => full_default_2
#   ones_like => full_default
#   ones_like_1 => full_default_1
#   ones_like_10 => full_default_11
#   ones_like_11 => full_default_12
#   ones_like_12 => full_default_13
#   ones_like_13 => full_default_14
#   ones_like_14 => full_default_16
#   ones_like_15 => full_default_17
#   ones_like_16 => full_default_18
#   ones_like_17 => full_default_19
#   ones_like_18 => full_default_21
#   ones_like_3 => full_default_4
#   ones_like_4 => full_default_5
#   ones_like_5 => full_default_6
#   ones_like_6 => full_default_7
#   ones_like_7 => full_default_8
#   ones_like_8 => full_default_9
#   ones_like_9 => full_default_10
#   or__4 => bitwise_or_4
#   or__5 => bitwise_or_5
#   p2 => pow_4
#   pick_x => le_11
#   prod => div_19
#   px_1 => sub_56
#   py_1 => sub_58
#   q2 => pow_5
#   r => sub_17
#   r_1 => sub_19
#   rm => mean
#   rm_1 => mean_1
#   sigmoid_2 => sigmoid_2
#   sigmoid_3 => sigmoid_3
#   sqrt => sqrt
#   sqrt_1 => sqrt_1
#   sqrt_2 => sqrt_2
#   sub_14 => sub_16
#   sub_19 => sub_21
#   sub_21 => sub_23
#   sub_23 => sub_25
#   sub_25 => sub_27
#   sub_27 => sub_29
#   sub_29 => sub_31
#   sub_31 => sub_33
#   sub_33 => sub_35
#   sub_35 => sub_37
#   sub_37 => sub_39
#   sub_39 => sub_41
#   sub_41 => sub_43
#   sub_43 => sub_45
#   sub_45 => sub_47
#   sub_47 => sub_49
#   sub_49 => sub_51
#   sub_51 => sub_53
#   sub_52 => sub_54
#   sub_57 => sub_59
#   sum_5 => sum_5
#   sum_6 => sum_6
#   sum_7 => sum_7
#   sum_8 => sum_8
#   sx => ldexp_2
#   sy => ldexp_3
#   targets => mul_20
#   total => clamp_min
#   truediv_10 => div_10
#   truediv_11 => div_11
#   truediv_12 => div_12
#   truediv_13 => div_13
#   truediv_14 => div_14
#   truediv_15 => div_15
#   truediv_16 => div_16
#   truediv_17 => div_17
#   truediv_18 => div_18
#   truediv_20 => div_20
#   truediv_21 => div_21
#   truediv_24 => div_24
#   truediv_25 => div_25
#   truediv_26 => div_26
#   truediv_27 => div_27
#   truediv_28 => div_28
#   truediv_29 => div_29
#   truediv_4 => div_4
#   truediv_5 => div_5
#   truediv_6 => div_6
#   truediv_7 => div_7
#   truediv_8 => div_8
#   truediv_9 => div_9
#   tx => div_22
#   ty => div_23
#   u => where_24
#   un => div_30
#   v => where_25
#   vn => div_31
#   where => where
#   where_1 => where_1
#   where_10 => where_10
#   where_11 => where_11
#   where_12 => where_12
#   where_13 => where_13
#   where_14 => where_14
#   where_15 => where_15
#   where_16 => where_16
#   where_18 => where_18
#   where_19 => where_19
#   where_2 => where_2
#   where_21 => where_21
#   where_22 => where_22
#   where_3 => where_3
#   where_4 => where_4
#   where_5 => where_5
#   where_6 => where_6
#   where_7 => where_7
#   where_8 => where_8
#   where_9 => where_9
#   xbar => add_17
#   xdeg => bitwise_or
#   xs => select_2, slice_7
#   ybar => add_18
#   ydeg => bitwise_or_1
#   ys => select_3, slice_8
#   zbar => mean_2
#   zero => full_default_3
#   zero_1 => full_default_15
#   zero_2 => full_default_20
# Graph fragment:
#   %cat : Tensor "f64[32768, 25, 2][50, 2, 1]cuda:0" = PlaceHolder[target=cat]
#   %mean_3 : Tensor "f64[32768, 5][5, 1]cuda:0" = PlaceHolder[target=mean_3]
#   %mean : Tensor "f64[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=mean]
#   %mean_4 : Tensor "f64[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=mean_4]
#   %ldexp : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=ldexp]
#   %ldexp_2 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=ldexp_2]
#   %mean_5 : Tensor "f64[32768, 5][5, 1]cuda:0" = PlaceHolder[target=mean_5]
#   %mean_1 : Tensor "f64[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=mean_1]
#   %ldexp_1 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=ldexp_1]
#   %mean_6 : Tensor "f64[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=mean_6]
#   %ldexp_3 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=ldexp_3]
#   %sum_3 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sum_3]
#   %sum_4 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sum_4]
#   %maximum_3 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=maximum_3]
#   %add_25 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_25]
#   %add_22 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_22]
#   %add_26 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_26]
#   %mul_31 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=mul_31]
#   %div_12 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_12]
#   %sub_28 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sub_28]
#   %div_14 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_14]
#   %where_7 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_7]
#   %sub_36 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sub_36]
#   %div_16 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_16]
#   %where_11 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_11]
#   %sub_44 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sub_44]
#   %div_18 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_18]
#   %where_15 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_15]
#   %sub_52 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sub_52]
#   %le_11 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=le_11]
#   %bitwise_or_3 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_3]
#   %bitwise_or_4 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_4]
#   %sum_7 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sum_7]
#   %bitwise_or_5 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_5]
#   %sum_8 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sum_8]
#   %maximum_5 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=maximum_5]
#   %buf14 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=buf14]
#   %where_20 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_20]
#   %where_23 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_23]
#   %slice_7 : Tensor "f64[32768, 5, 2][50, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, 1, 0, 9223372036854775807, 5), kwargs = {})
#   %select_2 : Tensor "f64[32768, 5][50, 10]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%slice_7, 2, 0), kwargs = {})
#   %slice_9 : Tensor "f64[32768, 1][50, 10]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select_2, 1, 0, 1), kwargs = {})
#   %sub_17 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_2, %slice_9), kwargs = {})
#   %mean : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_17, [1], True), kwargs = {})
#   %sub_18 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_17, %mean), kwargs = {})
#   %abs_3 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_18,), kwargs = {})
#   %amax : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_3, [1]), kwargs = {})
#   %frexp : [num_users=1] = call_function[target=torch.ops.aten.frexp.Tensor](args = (%amax,), kwargs = {})
#   %slice_8 : Tensor "f64[32768, 5, 2][50, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, 1, 0, 5), kwargs = {})
#   %select_3 : Tensor "f64[32768, 5][50, 2]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%slice_8, 2, 1), kwargs = {})
#   %slice_10 : Tensor "f64[32768, 1][50, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select_3, 1, 0, 1), kwargs = {})
#   %sub_19 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_3, %slice_10), kwargs = {})
#   %mean_1 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_19, [1], True), kwargs = {})
#   %sub_20 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_19, %mean_1), kwargs = {})
#   %abs_4 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_20,), kwargs = {})
#   %amax_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_4, [1]), kwargs = {})
#   %frexp_1 : [num_users=1] = call_function[target=torch.ops.aten.frexp.Tensor](args = (%amax_1,), kwargs = {})
#   %select : Tensor "f64[32768, 25][50, 2]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%cat, 2, 0), kwargs = {})
#   %sigmoid_2 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select,), kwargs = {})
#   %sub_16 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid_2), kwargs = {})
#   %sigmoid_3 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select,), kwargs = {})
#   %select_1 : Tensor "f64[32768, 25][50, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat, 2, 1), kwargs = {})
#   %mul_19 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_3, %select_1), kwargs = {})
#   %mul_20 : Tensor "f64[32768, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %mul_19), kwargs = {})
#   %mean_4 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mean_3, [1], True), kwargs = {})
#   %sub_56 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_3, %mean_4), kwargs = {})
#   %abs_7 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_56,), kwargs = {})
#   %amax_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_7, [1]), kwargs = {})
#   %frexp_2 : [num_users=1] = call_function[target=torch.ops.aten.frexp.Tensor](args = (%amax_2,), kwargs = {})
#   %mean_6 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mean_5, [1], True), kwargs = {})
#   %sub_58 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_5, %mean_6), kwargs = {})
#   %abs_8 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_58,), kwargs = {})
#   %amax_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_8, [1]), kwargs = {})
#   %frexp_3 : [num_users=1] = call_function[target=torch.ops.aten.frexp.Tensor](args = (%amax_3,), kwargs = {})
#   %mean_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_20, [1]), kwargs = {})
#   %full_default_15 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32768], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ldexp : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ldexp.Tensor](args = (%full_default, %getitem_1), kwargs = {})
#   %unsqueeze_13 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%ldexp, 1), kwargs = {})
#   %div_2 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_18, %unsqueeze_13), kwargs = {})
#   %full_default_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ldexp_2 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ldexp.Tensor](args = (%full_default_13, %getitem_5), kwargs = {})
#   %unsqueeze_15 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%ldexp_2, 1), kwargs = {})
#   %div_24 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_56, %unsqueeze_15), kwargs = {})
#   %mul_77 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %div_24), kwargs = {})
#   %sum_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_77, [1]), kwargs = {})
#   %select_8 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_2, 1, 0), kwargs = {})
#   %select_9 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_2, 1, -1), kwargs = {})
#   %eq_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_8, %select_9), kwargs = {})
#   %mul_23 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %sub_18), kwargs = {})
#   %sum_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_23, [1]), kwargs = {})
#   %eq_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_5, 0), kwargs = {})
#   %bitwise_or : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%eq_2, %eq_3), kwargs = {})
#   %full_default_2 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_4 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_2, 1, 0), kwargs = {})
#   %select_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%mean, 1, 0), kwargs = {})
#   %add_17 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %abs_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_17,), kwargs = {})
#   %select_6 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_3, 1, 0), kwargs = {})
#   %select_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%mean_1, 1, 0), kwargs = {})
#   %add_18 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_6, %select_7), kwargs = {})
#   %abs_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_18,), kwargs = {})
#   %maximum : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%abs_5, %abs_6), kwargs = {})
#   %maximum_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%full_default_2, %maximum), kwargs = {})
#   %mul_21 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %div_2), kwargs = {})
#   %sum_3 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_21, [1]), kwargs = {})
#   %div_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, 5), kwargs = {})
#   %sqrt : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%div_4,), kwargs = {})
#   %mul_25 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%ldexp, %sqrt), kwargs = {})
#   %full_default_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ldexp_1 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ldexp.Tensor](args = (%full_default_1, %getitem_3), kwargs = {})
#   %unsqueeze_14 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%ldexp_1, 1), kwargs = {})
#   %div_3 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_20, %unsqueeze_14), kwargs = {})
#   %mul_22 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %div_3), kwargs = {})
#   %sum_4 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_22, [1]), kwargs = {})
#   %div_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_4, 5), kwargs = {})
#   %sqrt_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%div_5,), kwargs = {})
#   %mul_26 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%ldexp_1, %sqrt_1), kwargs = {})
#   %maximum_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%mul_25, %mul_26), kwargs = {})
#   %maximum_3 : Tensor "f64[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum_1, %maximum_2), kwargs = {})
#   %div_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%full_default_2, %maximum_3), kwargs = {})
#   %pow_1 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_6, 2), kwargs = {})
#   %div_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_17, %maximum_3), kwargs = {})
#   %pow_2 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_7, 2), kwargs = {})
#   %add_19 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %div_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_18, %maximum_3), kwargs = {})
#   %pow_3 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_8, 2), kwargs = {})
#   %add_20 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %pow_3), kwargs = {})
#   %div_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_25, %maximum_3), kwargs = {})
#   %pow_4 : Tensor "f64[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_9, 2), kwargs = {})
#   %add_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %pow_4), kwargs = {})
#   %div_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_26, %maximum_3), kwargs = {})
#   %pow_5 : Tensor "f64[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_10, 2), kwargs = {})
#   %add_22 : Tensor "f64[32768][1]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %pow_5), kwargs = {})
#   %mul_34 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 3.0), kwargs = {})
#   %mul_35 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 2.0), kwargs = {})
#   %sub_23 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_34, %mul_35), kwargs = {})
#   %mul_36 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %add_22), kwargs = {})
#   %add_23 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_3), kwargs = {})
#   %mul_27 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_4, %add_23), kwargs = {})
#   %add_24 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %mul_28 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_5, %add_24), kwargs = {})
#   %add_25 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %mul_28), kwargs = {})
#   %mul_29 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_4, %pow_5), kwargs = {})
#   %add_26 : Tensor "f64[32768][1]cuda:0"[num_users=17] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %mul_29), kwargs = {})
#   %add_28 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %add_26), kwargs = {})
#   %gt : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_28, 0), kwargs = {})
#   %sub_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_22, %add_22), kwargs = {})
#   %mul_32 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %add_22), kwargs = {})
#   %add_27 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %add_26), kwargs = {})
#   %mul_33 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %add_22), kwargs = {})
#   %mul_30 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, %pow_4), kwargs = {})
#   %mul_31 : Tensor "f64[32768][1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %pow_5), kwargs = {})
#   %sub_22 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_33, %mul_31), kwargs = {})
#   %le : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_28, 0), kwargs = {})
#   %full_default_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default_4, %add_28), kwargs = {})
#   %div_11 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_22, %where), kwargs = {})
#   %full_default_3 : Tensor "f64[32768][1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([32768], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %div_11, %full_default_3), kwargs = {})
#   %sub_24 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_22, %where_1), kwargs = {})
#   %mul_39 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, 3.0), kwargs = {})
#   %mul_40 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 2.0), kwargs = {})
#   %sub_27 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_39, %mul_40), kwargs = {})
#   %mul_41 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %sub_24), kwargs = {})
#   %add_30 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %add_26), kwargs = {})
#   %gt_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_30, 0), kwargs = {})
#   %sub_25 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_24, %add_22), kwargs = {})
#   %mul_37 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %sub_24), kwargs = {})
#   %add_29 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %add_26), kwargs = {})
#   %mul_38 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sub_24), kwargs = {})
#   %sub_26 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_38, %mul_31), kwargs = {})
#   %le_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_30, 0), kwargs = {})
#   %full_default_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default_5, %add_30), kwargs = {})
#   %div_12 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_26, %where_2), kwargs = {})
#   %where_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %div_12, %full_default_3), kwargs = {})
#   %sub_28 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_24, %where_3), kwargs = {})
#   %mul_44 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, 3.0), kwargs = {})
#   %mul_45 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 2.0), kwargs = {})
#   %sub_31 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_44, %mul_45), kwargs = {})
#   %mul_46 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %sub_28), kwargs = {})
#   %add_32 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %add_26), kwargs = {})
#   %gt_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_32, 0), kwargs = {})
#   %sub_29 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_28, %add_22), kwargs = {})
#   %mul_42 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %sub_28), kwargs = {})
#   %add_31 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %add_26), kwargs = {})
#   %mul_43 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_31, %sub_28), kwargs = {})
#   %sub_30 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_43, %mul_31), kwargs = {})
#   %le_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_32, 0), kwargs = {})
#   %full_default_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %full_default_6, %add_32), kwargs = {})
#   %div_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_30, %where_4), kwargs = {})
#   %where_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %div_13, %full_default_3), kwargs = {})
#   %sub_32 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_28, %where_5), kwargs = {})
#   %mul_49 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, 3.0), kwargs = {})
#   %mul_50 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 2.0), kwargs = {})
#   %sub_35 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_49, %mul_50), kwargs = {})
#   %mul_51 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %sub_32), kwargs = {})
#   %add_34 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %add_26), kwargs = {})
#   %gt_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_34, 0), kwargs = {})
#   %sub_33 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_32, %add_22), kwargs = {})
#   %mul_47 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %sub_32), kwargs = {})
#   %add_33 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %add_26), kwargs = {})
#   %mul_48 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_33, %sub_32), kwargs = {})
#   %sub_34 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_48, %mul_31), kwargs = {})
#   %le_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_34, 0), kwargs = {})
#   %full_default_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default_7, %add_34), kwargs = {})
#   %div_14 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_34, %where_6), kwargs = {})
#   %where_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %div_14, %full_default_3), kwargs = {})
#   %sub_36 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_32, %where_7), kwargs = {})
#   %mul_54 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, 3.0), kwargs = {})
#   %mul_55 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 2.0), kwargs = {})
#   %sub_39 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_54, %mul_55), kwargs = {})
#   %mul_56 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %sub_36), kwargs = {})
#   %add_36 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %add_26), kwargs = {})
#   %gt_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_36, 0), kwargs = {})
#   %sub_37 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_36, %add_22), kwargs = {})
#   %mul_52 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %sub_36), kwargs = {})
#   %add_35 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_52, %add_26), kwargs = {})
#   %mul_53 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_35, %sub_36), kwargs = {})
#   %sub_38 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_53, %mul_31), kwargs = {})
#   %le_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_36, 0), kwargs = {})
#   %full_default_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %full_default_8, %add_36), kwargs = {})
#   %div_15 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_38, %where_8), kwargs = {})
#   %where_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %div_15, %full_default_3), kwargs = {})
#   %sub_40 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_36, %where_9), kwargs = {})
#   %mul_59 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, 3.0), kwargs = {})
#   %mul_60 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 2.0), kwargs = {})
#   %sub_43 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_59, %mul_60), kwargs = {})
#   %mul_61 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %sub_40), kwargs = {})
#   %add_38 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %add_26), kwargs = {})
#   %gt_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_38, 0), kwargs = {})
#   %sub_41 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_40, %add_22), kwargs = {})
#   %mul_57 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %sub_40), kwargs = {})
#   %add_37 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_57, %add_26), kwargs = {})
#   %mul_58 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %sub_40), kwargs = {})
#   %sub_42 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_58, %mul_31), kwargs = {})
#   %le_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_38, 0), kwargs = {})
#   %full_default_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %full_default_9, %add_38), kwargs = {})
#   %div_16 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_42, %where_10), kwargs = {})
#   %where_11 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %div_16, %full_default_3), kwargs = {})
#   %sub_44 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_40, %where_11), kwargs = {})
#   %mul_64 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, 3.0), kwargs = {})
#   %mul_65 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 2.0), kwargs = {})
#   %sub_47 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_64, %mul_65), kwargs = {})
#   %mul_66 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %sub_44), kwargs = {})
#   %add_40 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, %add_26), kwargs = {})
#   %gt_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_40, 0), kwargs = {})
#   %sub_45 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_44, %add_22), kwargs = {})
#   %mul_62 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %sub_44), kwargs = {})
#   %add_39 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %add_26), kwargs = {})
#   %mul_63 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_39, %sub_44), kwargs = {})
#   %sub_46 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_63, %mul_31), kwargs = {})
#   %le_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_40, 0), kwargs = {})
#   %full_default_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %full_default_10, %add_40), kwargs = {})
#   %div_17 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_46, %where_12), kwargs = {})
#   %where_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %div_17, %full_default_3), kwargs = {})
#   %sub_48 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_44, %where_13), kwargs = {})
#   %mul_69 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, 3.0), kwargs = {})
#   %mul_70 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 2.0), kwargs = {})
#   %sub_51 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_69, %mul_70), kwargs = {})
#   %mul_71 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %sub_48), kwargs = {})
#   %add_42 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %add_26), kwargs = {})
#   %gt_7 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_42, 0), kwargs = {})
#   %sub_49 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_48, %add_22), kwargs = {})
#   %mul_67 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %sub_48), kwargs = {})
#   %add_41 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_67, %add_26), kwargs = {})
#   %mul_68 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, %sub_48), kwargs = {})
#   %sub_50 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_68, %mul_31), kwargs = {})
#   %le_7 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_42, 0), kwargs = {})
#   %full_default_11 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_14 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %full_default_11, %add_42), kwargs = {})
#   %div_18 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_50, %where_14), kwargs = {})
#   %where_15 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %div_18, %full_default_3), kwargs = {})
#   %sub_52 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_48, %where_15), kwargs = {})
#   %div_19 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_31, %sub_52), kwargs = {})
#   %sub_53 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_26, %div_19), kwargs = {})
#   %div_20 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_53, %sub_52), kwargs = {})
#   %clamp_min : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_20, 0), kwargs = {})
#   %mul_72 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min, %clamp_min), kwargs = {})
#   %mul_73 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_19, 4.0), kwargs = {})
#   %sub_54 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_72, %mul_73), kwargs = {})
#   %clamp_min_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_54, 0), kwargs = {})
#   %sqrt_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_1,), kwargs = {})
#   %add_43 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min, %sqrt_2), kwargs = {})
#   %mul_74 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, 0.5), kwargs = {})
#   %mul_75 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, 3.0814879110195774e-29), kwargs = {})
#   %le_9 : Tensor "b8[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%mul_74, %mul_75), kwargs = {})
#   %gt_8 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_74, 0), kwargs = {})
#   %le_8 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%mul_74, 0), kwargs = {})
#   %full_default_12 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_16 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default_12, %mul_74), kwargs = {})
#   %div_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_19, %where_16), kwargs = {})
#   %where_17 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %div_21, %full_default_3), kwargs = {})
#   %mul_76 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, 3.0814879110195774e-29), kwargs = {})
#   %le_10 : Tensor "b8[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%where_17, %mul_76), kwargs = {})
#   %add_44 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %div_22 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_4, %add_44), kwargs = {})
#   %add_45 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_3), kwargs = {})
#   %div_23 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_5, %add_45), kwargs = {})
#   %le_11 : Tensor "b8[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%div_22, %div_23), kwargs = {})
#   %bitwise_and : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%le_10, %le_11), kwargs = {})
#   %bitwise_or_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_9, %bitwise_and), kwargs = {})
#   %bitwise_or_4 : Tensor "b8[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or, %bitwise_or_2), kwargs = {})
#   %full_default_16 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_18 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %full_default_16, %sum_3), kwargs = {})
#   %div_25 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_7, %where_18), kwargs = {})
#   %mul_78 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_25, %ldexp_2), kwargs = {})
#   %full_default_17 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_19 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %full_default_17, %ldexp), kwargs = {})
#   %div_26 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_78, %where_19), kwargs = {})
#   %where_20 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %full_default_15, %div_26), kwargs = {})
#   %mul_81 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_20, %add_17), kwargs = {})
#   %sub_59 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_2, %mul_81), kwargs = {})
#   %full_default_14 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ldexp_3 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ldexp.Tensor](args = (%full_default_14, %getitem_7), kwargs = {})
#   %unsqueeze_16 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%ldexp_3, 1), kwargs = {})
#   %div_27 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_58, %unsqueeze_16), kwargs = {})
#   %mul_79 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %div_27), kwargs = {})
#   %sum_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_79, [1]), kwargs = {})
#   %select_10 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_3, 1, 0), kwargs = {})
#   %select_11 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_3, 1, -1), kwargs = {})
#   %eq_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_10, %select_11), kwargs = {})
#   %mul_24 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %sub_20), kwargs = {})
#   %sum_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_24, [1]), kwargs = {})
#   %eq_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_6, 0), kwargs = {})
#   %bitwise_or_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%eq_4, %eq_5), kwargs = {})
#   %bitwise_not : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%le_11,), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%le_10, %bitwise_not), kwargs = {})
#   %bitwise_or_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_9, %bitwise_and_1), kwargs = {})
#   %bitwise_or_5 : Tensor "b8[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or_1, %bitwise_or_3), kwargs = {})
#   %full_default_18 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %full_default_18, %sum_4), kwargs = {})
#   %div_28 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_8, %where_21), kwargs = {})
#   %mul_80 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_28, %ldexp_3), kwargs = {})
#   %full_default_19 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_22 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %full_default_19, %ldexp_1), kwargs = {})
#   %div_29 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_80, %where_22), kwargs = {})
#   %where_23 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %full_default_15, %div_29), kwargs = {})
#   %mul_82 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_23, %add_18), kwargs = {})
#   %sub_60 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_59, %mul_82), kwargs = {})
#   %full_default_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_20 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32768], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_24 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %add_17, %full_default_20), kwargs = {})
#   %abs_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where_24,), kwargs = {})
#   %where_25 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %add_18, %full_default_20), kwargs = {})
#   %abs_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where_25,), kwargs = {})
#   %maximum_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%abs_9, %abs_10), kwargs = {})
#   %maximum_5 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.maximum.default](args = (%full_default_21, %maximum_4), kwargs = {})
#   %div_32 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_60, %maximum_5), kwargs = {})
#   %reciprocal : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%maximum_5,), kwargs = {})
#   %mul_83 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %mul_84 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, %mul_83), kwargs = {})
#   %div_30 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_24, %maximum_5), kwargs = {})
#   %mul_85 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_30, %div_30), kwargs = {})
#   %add_46 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %mul_85), kwargs = {})
#   %div_31 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_25, %maximum_5), kwargs = {})
#   %mul_86 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_31, %div_31), kwargs = {})
#   %add_47 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %mul_86), kwargs = {})
#   return %buf14,%mean,%mean_4,%ldexp,%ldexp_2,%sum_7,%sum_3,%mean_6,%mean_1,%ldexp_1,%sum_4,%ldexp_3,%sum_8,%maximum_3,%add_22,%add_25,%le_11,%add_26,%mul_31,%div_12,%sub_28,%div_14,%where_7,%sub_36,%div_16,%where_11,%sub_44,%div_18,%where_15,%sub_52,%bitwise_or_4,%bitwise_or_3,%bitwise_or_5,%where_20,%where_23,%maximum_5,%add_47,%div_32
triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_rsub_select_sigmoid_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_4 = async_compile.triton('triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_rsub_select_sigmoid_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_out_ptr1': '*fp64', 'in_out_ptr7': '*i1', 'in_out_ptr8': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'out_ptr0': '*fp64', 'out_ptr4': '*fp64', 'out_ptr12': '*i1', 'out_ptr13': '*fp64', 'out_ptr14': '*fp64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_rsub_select_sigmoid_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr7', 'in_out_ptr8'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 22, 'num_store': 9, 'num_reduction': 1, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3801088, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_rsub_select_sigmoid_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_4(in_out_ptr0, in_out_ptr1, in_out_ptr7, in_out_ptr8, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr4, out_ptr12, out_ptr13, out_ptr14, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 25
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*r0_1 + 50*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr0 + (1 + 2*r0_1 + 50*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr0 + (50*x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (10 + 50*x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (20 + 50*x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (30 + 50*x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (40 + 50*x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (5*x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr1 + (1 + 5*x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (2 + 5*x0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (3 + 5*x0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr1 + (4 + 5*x0), None, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr2 + (5*x0), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr2 + (1 + 5*x0), None, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr2 + (2 + 5*x0), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr2 + (3 + 5*x0), None, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr2 + (4 + 5*x0), None, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr0 + (1 + 50*x0), None, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr0 + (3 + 50*x0), None, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr0 + (5 + 50*x0), None, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr0 + (7 + 50*x0), None, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr0 + (9 + 50*x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 1, tl.int32)
    tmp2 = -tmp0
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tmp1 + tmp3
    tmp5 = (tmp1 / tmp4)
    tmp6 = tl.full([1, 1], 1.0, tl.float64)
    tmp7 = tmp6 - tmp5
    tmp9 = tmp5 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float64)
    tmp16 = tmp15 - tmp15
    tmp18 = tmp17 - tmp15
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20 - tmp15
    tmp22 = tmp19 + tmp21
    tmp24 = tmp23 - tmp15
    tmp25 = tmp22 + tmp24
    tmp27 = tmp26 - tmp15
    tmp28 = tmp25 + tmp27
    tmp29 = tl.full([1, 1], 5.0, tl.float64)
    tmp30 = (tmp28 / tmp29)
    tmp33 = tmp31 + tmp32
    tmp35 = tmp33 + tmp34
    tmp37 = tmp35 + tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = (tmp39 / tmp29)
    tmp41 = tmp16 - tmp30
    tmp42 = tl_math.abs(tmp41)
    tmp43 = tmp18 - tmp30
    tmp44 = tl_math.abs(tmp43)
    tmp45 = triton_helpers.maximum(tmp42, tmp44)
    tmp46 = tmp21 - tmp30
    tmp47 = tl_math.abs(tmp46)
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp24 - tmp30
    tmp50 = tl_math.abs(tmp49)
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tmp52 = tmp27 - tmp30
    tmp53 = tl_math.abs(tmp52)
    tmp54 = triton_helpers.maximum(tmp51, tmp53)
    tmp55, tmp56 = triton_helpers.frexp(tmp54)
    tmp57 = libdevice.ldexp(tmp6, tmp56.to(tl.int32))
    tmp58 = tmp31 - tmp40
    tmp59 = tl_math.abs(tmp58)
    tmp60 = tmp32 - tmp40
    tmp61 = tl_math.abs(tmp60)
    tmp62 = triton_helpers.maximum(tmp59, tmp61)
    tmp63 = tmp34 - tmp40
    tmp64 = tl_math.abs(tmp63)
    tmp65 = triton_helpers.maximum(tmp62, tmp64)
    tmp66 = tmp36 - tmp40
    tmp67 = tl_math.abs(tmp66)
    tmp68 = triton_helpers.maximum(tmp65, tmp67)
    tmp69 = tmp38 - tmp40
    tmp70 = tl_math.abs(tmp69)
    tmp71 = triton_helpers.maximum(tmp68, tmp70)
    tmp72, tmp73 = triton_helpers.frexp(tmp71)
    tmp74 = libdevice.ldexp(tmp6, tmp73.to(tl.int32))
    tmp75 = (tmp41 / tmp57)
    tmp76 = (tmp58 / tmp74)
    tmp77 = tmp75 * tmp76
    tmp78 = (tmp43 / tmp57)
    tmp79 = (tmp60 / tmp74)
    tmp80 = tmp78 * tmp79
    tmp81 = tmp77 + tmp80
    tmp82 = (tmp46 / tmp57)
    tmp83 = (tmp63 / tmp74)
    tmp84 = tmp82 * tmp83
    tmp85 = tmp81 + tmp84
    tmp86 = (tmp49 / tmp57)
    tmp87 = (tmp66 / tmp74)
    tmp88 = tmp86 * tmp87
    tmp89 = tmp85 + tmp88
    tmp90 = (tmp52 / tmp57)
    tmp91 = (tmp69 / tmp74)
    tmp92 = tmp90 * tmp91
    tmp93 = tmp89 + tmp92
    tmp94 = tmp75 * tmp75
    tmp95 = tmp78 * tmp78
    tmp96 = tmp94 + tmp95
    tmp97 = tmp82 * tmp82
    tmp98 = tmp96 + tmp97
    tmp99 = tmp86 * tmp86
    tmp100 = tmp98 + tmp99
    tmp101 = tmp90 * tmp90
    tmp102 = tmp100 + tmp101
    tmp105 = tmp103 + tmp104
    tmp107 = tmp105 + tmp106
    tmp109 = tmp107 + tmp108
    tmp111 = tmp109 + tmp110
    tmp112 = (tmp111 / tmp29)
    tmp114 = tmp113 - tmp113
    tmp116 = tmp115 - tmp113
    tmp117 = tmp114 + tmp116
    tmp119 = tmp118 - tmp113
    tmp120 = tmp117 + tmp119
    tmp122 = tmp121 - tmp113
    tmp123 = tmp120 + tmp122
    tmp125 = tmp124 - tmp113
    tmp126 = tmp123 + tmp125
    tmp127 = (tmp126 / tmp29)
    tmp128 = tmp114 - tmp127
    tmp129 = tl_math.abs(tmp128)
    tmp130 = tmp116 - tmp127
    tmp131 = tl_math.abs(tmp130)
    tmp132 = triton_helpers.maximum(tmp129, tmp131)
    tmp133 = tmp119 - tmp127
    tmp134 = tl_math.abs(tmp133)
    tmp135 = triton_helpers.maximum(tmp132, tmp134)
    tmp136 = tmp122 - tmp127
    tmp137 = tl_math.abs(tmp136)
    tmp138 = triton_helpers.maximum(tmp135, tmp137)
    tmp139 = tmp125 - tmp127
    tmp140 = tl_math.abs(tmp139)
    tmp141 = triton_helpers.maximum(tmp138, tmp140)
    tmp142, tmp143 = triton_helpers.frexp(tmp141)
    tmp144 = libdevice.ldexp(tmp6, tmp143.to(tl.int32))
    tmp145 = (tmp128 / tmp144)
    tmp146 = tmp145 * tmp145
    tmp147 = (tmp130 / tmp144)
    tmp148 = tmp147 * tmp147
    tmp149 = tmp146 + tmp148
    tmp150 = (tmp133 / tmp144)
    tmp151 = tmp150 * tmp150
    tmp152 = tmp149 + tmp151
    tmp153 = (tmp136 / tmp144)
    tmp154 = tmp153 * tmp153
    tmp155 = tmp152 + tmp154
    tmp156 = (tmp139 / tmp144)
    tmp157 = tmp156 * tmp156
    tmp158 = tmp155 + tmp157
    tmp159 = tmp103 - tmp112
    tmp160 = tl_math.abs(tmp159)
    tmp161 = tmp104 - tmp112
    tmp162 = tl_math.abs(tmp161)
    tmp163 = triton_helpers.maximum(tmp160, tmp162)
    tmp164 = tmp106 - tmp112
    tmp165 = tl_math.abs(tmp164)
    tmp166 = triton_helpers.maximum(tmp163, tmp165)
    tmp167 = tmp108 - tmp112
    tmp168 = tl_math.abs(tmp167)
    tmp169 = triton_helpers.maximum(tmp166, tmp168)
    tmp170 = tmp110 - tmp112
    tmp171 = tl_math.abs(tmp170)
    tmp172 = triton_helpers.maximum(tmp169, tmp171)
    tmp173, tmp174 = triton_helpers.frexp(tmp172)
    tmp175 = libdevice.ldexp(tmp6, tmp174.to(tl.int32))
    tmp176 = (tmp159 / tmp175)
    tmp177 = tmp145 * tmp176
    tmp178 = (tmp161 / tmp175)
    tmp179 = tmp147 * tmp178
    tmp180 = tmp177 + tmp179
    tmp181 = (tmp164 / tmp175)
    tmp182 = tmp150 * tmp181
    tmp183 = tmp180 + tmp182
    tmp184 = (tmp167 / tmp175)
    tmp185 = tmp153 * tmp184
    tmp186 = tmp183 + tmp185
    tmp187 = (tmp170 / tmp175)
    tmp188 = tmp156 * tmp187
    tmp189 = tmp186 + tmp188
    tmp190 = tmp15 + tmp30
    tmp191 = tl_math.abs(tmp190)
    tmp192 = tmp113 + tmp127
    tmp193 = tl_math.abs(tmp192)
    tmp194 = triton_helpers.maximum(tmp191, tmp193)
    tmp195 = triton_helpers.maximum(tmp6, tmp194)
    tmp196 = tl.full([1, 1], 0.2, tl.float64)
    tmp197 = tmp102 * tmp196
    tmp198 = libdevice.sqrt(tmp197)
    tmp199 = tmp57 * tmp198
    tmp200 = tmp158 * tmp196
    tmp201 = libdevice.sqrt(tmp200)
    tmp202 = tmp144 * tmp201
    tmp203 = triton_helpers.maximum(tmp199, tmp202)
    tmp204 = triton_helpers.maximum(tmp195, tmp203)
    tmp205 = (tmp6 / tmp204)
    tmp206 = tmp205 * tmp205
    tmp207 = (tmp190 / tmp204)
    tmp208 = tmp207 * tmp207
    tmp209 = tmp206 + tmp208
    tmp210 = (tmp192 / tmp204)
    tmp211 = tmp210 * tmp210
    tmp212 = tmp209 + tmp211
    tmp213 = (tmp199 / tmp204)
    tmp214 = tmp213 * tmp213
    tmp215 = tmp212 + tmp214
    tmp216 = (tmp202 / tmp204)
    tmp217 = tmp216 * tmp216
    tmp218 = tmp215 + tmp217
    tmp219 = tmp206 + tmp211
    tmp220 = tmp214 * tmp219
    tmp221 = tmp217 * tmp209
    tmp222 = tmp220 + tmp221
    tmp223 = (tmp214 / tmp209)
    tmp224 = (tmp217 / tmp219)
    tmp225 = tmp223 <= tmp224
    tmp226 = tmp214 * tmp217
    tmp227 = tmp222 + tmp226
    tmp228 = tmp206 * tmp214
    tmp229 = tmp228 * tmp217
    tmp230 = tl.full([1, 1], 3.0, tl.float64)
    tmp231 = tmp218 * tmp230
    tmp232 = tl.full([1, 1], 2.0, tl.float64)
    tmp233 = tmp218 * tmp232
    tmp234 = tmp231 - tmp233
    tmp235 = tmp234 * tmp218
    tmp236 = tmp235 + tmp227
    tmp237 = tl.full([1, 1], 0.0, tl.float64)
    tmp238 = tmp236 > tmp237
    tmp239 = tmp218 - tmp218
    tmp240 = tmp239 * tmp218
    tmp241 = tmp240 + tmp227
    tmp242 = tmp241 * tmp218
    tmp243 = tmp242 - tmp229
    tmp244 = tmp236 <= tmp237
    tmp245 = tl.where(tmp244, tmp6, tmp236)
    tmp246 = (tmp243 / tmp245)
    tmp247 = tl.where(tmp238, tmp246, tmp237)
    tmp248 = tmp218 - tmp247
    tmp249 = tmp248 - tmp218
    tmp250 = tmp249 * tmp248
    tmp251 = tmp250 + tmp227
    tmp252 = tmp251 * tmp248
    tmp253 = tmp252 - tmp229
    tmp254 = tmp248 * tmp230
    tmp255 = tmp254 - tmp233
    tmp256 = tmp255 * tmp248
    tmp257 = tmp256 + tmp227
    tmp258 = tmp257 <= tmp237
    tmp259 = tl.where(tmp258, tmp6, tmp257)
    tmp260 = (tmp253 / tmp259)
    tmp261 = tmp257 > tmp237
    tmp262 = tl.where(tmp261, tmp260, tmp237)
    tmp263 = tmp248 - tmp262
    tmp264 = tmp263 * tmp230
    tmp265 = tmp264 - tmp233
    tmp266 = tmp265 * tmp263
    tmp267 = tmp266 + tmp227
    tmp268 = tmp267 > tmp237
    tmp269 = tmp263 - tmp218
    tmp270 = tmp269 * tmp263
    tmp271 = tmp270 + tmp227
    tmp272 = tmp271 * tmp263
    tmp273 = tmp272 - tmp229
    tmp274 = tmp267 <= tmp237
    tmp275 = tl.where(tmp274, tmp6, tmp267)
    tmp276 = (tmp273 / tmp275)
    tmp277 = tl.where(tmp268, tmp276, tmp237)
    tmp278 = tmp263 - tmp277
    tmp279 = tmp278 - tmp218
    tmp280 = tmp279 * tmp278
    tmp281 = tmp280 + tmp227
    tmp282 = tmp281 * tmp278
    tmp283 = tmp282 - tmp229
    tmp284 = tmp278 * tmp230
    tmp285 = tmp284 - tmp233
    tmp286 = tmp285 * tmp278
    tmp287 = tmp286 + tmp227
    tmp288 = tmp287 <= tmp237
    tmp289 = tl.where(tmp288, tmp6, tmp287)
    tmp290 = (tmp283 / tmp289)
    tmp291 = tmp287 > tmp237
    tmp292 = tl.where(tmp291, tmp290, tmp237)
    tmp293 = tmp278 - tmp292
    tmp294 = tmp293 * tmp230
    tmp295 = tmp294 - tmp233
    tmp296 = tmp295 * tmp293
    tmp297 = tmp296 + tmp227
    tmp298 = tmp297 > tmp237
    tmp299 = tmp293 - tmp218
    tmp300 = tmp299 * tmp293
    tmp301 = tmp300 + tmp227
    tmp302 = tmp301 * tmp293
    tmp303 = tmp302 - tmp229
    tmp304 = tmp297 <= tmp237
    tmp305 = tl.where(tmp304, tmp6, tmp297)
    tmp306 = (tmp303 / tmp305)
    tmp307 = tl.where(tmp298, tmp306, tmp237)
    tmp308 = tmp293 - tmp307
    tmp309 = tmp308 - tmp218
    tmp310 = tmp309 * tmp308
    tmp311 = tmp310 + tmp227
    tmp312 = tmp311 * tmp308
    tmp313 = tmp312 - tmp229
    tmp314 = tmp308 * tmp230
    tmp315 = tmp314 - tmp233
    tmp316 = tmp315 * tmp308
    tmp317 = tmp316 + tmp227
    tmp318 = tmp317 <= tmp237
    tmp319 = tl.where(tmp318, tmp6, tmp317)
    tmp320 = (tmp313 / tmp319)
    tmp321 = tmp317 > tmp237
    tmp322 = tl.where(tmp321, tmp320, tmp237)
    tmp323 = tmp308 - tmp322
    tmp324 = tmp323 * tmp230
    tmp325 = tmp324 - tmp233
    tmp326 = tmp325 * tmp323
    tmp327 = tmp326 + tmp227
    tmp328 = tmp327 > tmp237
    tmp329 = tmp323 - tmp218
    tmp330 = tmp329 * tmp323
    tmp331 = tmp330 + tmp227
    tmp332 = tmp331 * tmp323
    tmp333 = tmp332 - tmp229
    tmp334 = tmp327 <= tmp237
    tmp335 = tl.where(tmp334, tmp6, tmp327)
    tmp336 = (tmp333 / tmp335)
    tmp337 = tl.where(tmp328, tmp336, tmp237)
    tmp338 = tmp323 - tmp337
    tmp339 = tmp338 - tmp218
    tmp340 = tmp339 * tmp338
    tmp341 = tmp340 + tmp227
    tmp342 = tmp341 * tmp338
    tmp343 = tmp342 - tmp229
    tmp344 = tmp338 * tmp230
    tmp345 = tmp344 - tmp233
    tmp346 = tmp345 * tmp338
    tmp347 = tmp346 + tmp227
    tmp348 = tmp347 <= tmp237
    tmp349 = tl.where(tmp348, tmp6, tmp347)
    tmp350 = (tmp343 / tmp349)
    tmp351 = tmp347 > tmp237
    tmp352 = tl.where(tmp351, tmp350, tmp237)
    tmp353 = tmp338 - tmp352
    tmp354 = tmp15 == tmp26
    tmp355 = tmp41 * tmp41
    tmp356 = tmp43 * tmp43
    tmp357 = tmp355 + tmp356
    tmp358 = tmp46 * tmp46
    tmp359 = tmp357 + tmp358
    tmp360 = tmp49 * tmp49
    tmp361 = tmp359 + tmp360
    tmp362 = tmp52 * tmp52
    tmp363 = tmp361 + tmp362
    tmp364 = tmp363 == tmp237
    tmp365 = tmp354 | tmp364
    tmp366 = (tmp229 / tmp353)
    tmp367 = tmp227 - tmp366
    tmp368 = (tmp367 / tmp353)
    tmp369 = triton_helpers.maximum(tmp368, tmp237)
    tmp370 = tmp369 * tmp369
    tmp371 = tl.full([1, 1], 4.0, tl.float64)
    tmp372 = tmp366 * tmp371
    tmp373 = tmp370 - tmp372
    tmp374 = triton_helpers.maximum(tmp373, tmp237)
    tmp375 = libdevice.sqrt(tmp374)
    tmp376 = tmp369 + tmp375
    tmp377 = tl.full([1, 1], 0.5, tl.float64)
    tmp378 = tmp376 * tmp377
    tmp379 = tl.full([1, 1], 3.0814879110195774e-29, tl.float64)
    tmp380 = tmp353 * tmp379
    tmp381 = tmp378 <= tmp380
    tmp382 = tmp378 > tmp237
    tmp383 = tmp378 <= tmp237
    tmp384 = tl.where(tmp383, tmp6, tmp378)
    tmp385 = (tmp366 / tmp384)
    tmp386 = tl.where(tmp382, tmp385, tmp237)
    tmp387 = tmp386 <= tmp380
    tmp388 = tmp387 & tmp225
    tmp389 = tmp381 | tmp388
    tmp390 = tmp365 | tmp389
    tmp391 = tmp225 == 0
    tmp392 = tmp387 & tmp391
    tmp393 = tmp381 | tmp392
    tmp394 = tmp113 == tmp124
    tmp395 = tmp128 * tmp128
    tmp396 = tmp130 * tmp130
    tmp397 = tmp395 + tmp396
    tmp398 = tmp133 * tmp133
    tmp399 = tmp397 + tmp398
    tmp400 = tmp136 * tmp136
    tmp401 = tmp399 + tmp400
    tmp402 = tmp139 * tmp139
    tmp403 = tmp401 + tmp402
    tmp404 = tmp403 == tmp237
    tmp405 = tmp394 | tmp404
    tmp406 = tmp405 | tmp393
    tmp407 = tl.where(tmp390, tmp6, tmp102)
    tmp408 = (tmp93 / tmp407)
    tmp409 = tmp408 * tmp74
    tmp410 = tl.where(tmp390, tmp6, tmp57)
    tmp411 = (tmp409 / tmp410)
    tmp412 = tl.where(tmp390, tmp237, tmp411)
    tmp413 = tl.where(tmp406, tmp6, tmp158)
    tmp414 = (tmp189 / tmp413)
    tmp415 = tmp414 * tmp175
    tmp416 = tl.where(tmp406, tmp6, tmp144)
    tmp417 = (tmp415 / tmp416)
    tmp418 = tl.where(tmp406, tmp237, tmp417)
    tmp419 = tl.where(tmp390, tmp190, tmp237)
    tmp420 = tl_math.abs(tmp419)
    tmp421 = tl.where(tmp406, tmp192, tmp237)
    tmp422 = tl_math.abs(tmp421)
    tmp423 = triton_helpers.maximum(tmp420, tmp422)
    tmp424 = triton_helpers.maximum(tmp6, tmp423)
    tmp425 = (tmp1 / tmp424)
    tmp426 = tmp425 * tmp6
    tmp427 = tmp426 * tmp426
    tmp428 = (tmp419 / tmp424)
    tmp429 = tmp428 * tmp428
    tmp430 = tmp427 + tmp429
    tmp431 = (tmp421 / tmp424)
    tmp432 = tmp431 * tmp431
    tmp433 = tmp430 + tmp432
    tmp434 = tl.full([1, 1], 25.0, tl.float64)
    tmp435 = (tmp14 / tmp434)
    tmp436 = tmp412 * tmp190
    tmp437 = tmp435 - tmp436
    tmp438 = tmp418 * tmp192
    tmp439 = tmp437 - tmp438
    tmp440 = (tmp439 / tmp424)
    tl.store(out_ptr0 + (x0), tmp30, None)
    tl.store(out_ptr4 + (x0), tmp127, None)
    tl.store(out_ptr12 + (x0), tmp390, None)
    tl.debug_barrier()
    tl.store(in_out_ptr7 + (x0), tmp406, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp412, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp418, None)
    tl.store(out_ptr13 + (x0), tmp424, None)
    tl.store(out_ptr14 + (x0), tmp433, None)
    tl.debug_barrier()
    tl.store(in_out_ptr8 + (x0), tmp440, None)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/ay/caylr5t426uacr4fkl4fnba2g3tutmjdy2ejeikqgkenufjysm2l.py
# Topologically Sorted Source Nodes: [xs, ys, getitem_10, getitem_11, xbar, getitem_13, getitem_14, ybar, zero_2, u, v, inv, mul_78, un, vn, C, coef, mul_79, truediv_35, A, mul_80, truediv_36, B], Original ATen: [aten.slice, aten.select, aten.add, aten.zeros_like, aten.where, aten.reciprocal, aten.mul, aten.div, aten.stack]
# Source node to ATen node mapping:
#   A => where_26
#   B => where_27
#   C => div_33
#   coef => cat_1, unsqueeze_17, unsqueeze_18, unsqueeze_19
#   getitem_10 => select_4
#   getitem_11 => select_5
#   getitem_13 => select_6
#   getitem_14 => select_7
#   inv => mul_83, reciprocal
#   mul_78 => mul_87
#   mul_79 => mul_88
#   mul_80 => mul_89
#   truediv_35 => div_34
#   truediv_36 => div_35
#   u => where_24
#   un => div_30
#   v => where_25
#   vn => div_31
#   xbar => add_17
#   xs => select_2, slice_7
#   ybar => add_18
#   ys => select_3, slice_8
#   zero_2 => full_default_20
# Graph fragment:
#   %div_32 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_32]
#   %maximum_5 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=maximum_5]
#   %add_47 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_47]
#   %bitwise_or_4 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_4]
#   %cat : Tensor "f64[32768, 25, 2][50, 2, 1]cuda:0" = PlaceHolder[target=cat]
#   %mean : Tensor "f64[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=mean]
#   %where_20 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_20]
#   %bitwise_or_5 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_5]
#   %mean_1 : Tensor "f64[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=mean_1]
#   %where_23 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_23]
#   %slice_7 : Tensor "f64[32768, 5, 2][50, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, 1, 0, 9223372036854775807, 5), kwargs = {})
#   %select_2 : Tensor "f64[32768, 5][50, 10]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%slice_7, 2, 0), kwargs = {})
#   %slice_8 : Tensor "f64[32768, 5, 2][50, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, 1, 0, 5), kwargs = {})
#   %select_3 : Tensor "f64[32768, 5][50, 2]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%slice_8, 2, 1), kwargs = {})
#   %select_4 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_2, 1, 0), kwargs = {})
#   %select_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%mean, 1, 0), kwargs = {})
#   %add_17 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %select_6 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_3, 1, 0), kwargs = {})
#   %select_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%mean_1, 1, 0), kwargs = {})
#   %add_18 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_6, %select_7), kwargs = {})
#   %full_default_20 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32768], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_24 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %add_17, %full_default_20), kwargs = {})
#   %where_25 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %add_18, %full_default_20), kwargs = {})
#   %reciprocal : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%maximum_5,), kwargs = {})
#   %mul_83 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %mul_87 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_32, %mul_83), kwargs = {})
#   %div_30 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_24, %maximum_5), kwargs = {})
#   %div_31 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_25, %maximum_5), kwargs = {})
#   %div_33 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_87, %add_47), kwargs = {})
#   %unsqueeze_17 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div_33, 1), kwargs = {})
#   %mul_88 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_32, %div_30), kwargs = {})
#   %div_34 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_88, %add_47), kwargs = {})
#   %where_26 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %div_34, %where_20), kwargs = {})
#   %unsqueeze_18 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where_26, 1), kwargs = {})
#   %mul_89 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_32, %div_31), kwargs = {})
#   %div_35 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_89, %add_47), kwargs = {})
#   %where_27 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %div_35, %where_23), kwargs = {})
#   %unsqueeze_19 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where_27, 1), kwargs = {})
#   %cat_1 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_17, %unsqueeze_18, %unsqueeze_19], 1), kwargs = {})
#   return %cat_1
triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_5 = async_compile.triton('triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*i1', 'in_ptr4': '*fp64', 'in_ptr5': '*fp64', 'in_ptr6': '*fp64', 'in_ptr7': '*i1', 'in_ptr8': '*fp64', 'in_ptr9': '*fp64', 'out_ptr0': '*fp64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 17, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1900544, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tl.full([1, 1], 1.0, tl.float64)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 * tmp10
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = (tmp11 / tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1, 1], 2, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp21 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to(50*y0, [YBLOCK, XBLOCK])), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full([1, 1], 0.0, tl.float64)
    tmp26 = tl.where(tmp20, tmp24, tmp25)
    tmp27 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = (tmp26 / tmp27)
    tmp29 = tmp21 * tmp28
    tmp30 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = (tmp29 / tmp30)
    tmp32 = tl.load(in_ptr6 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp20, tmp31, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tmp0 >= tmp17
    tmp37 = tl.full([1, 1], 3, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr7 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp36 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp40 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr4 + (tl.broadcast_to(1 + 50*y0, [YBLOCK, XBLOCK])), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr8 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1, 1], 0.0, tl.float64)
    tmp45 = tl.where(tmp39, tmp43, tmp44)
    tmp46 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = (tmp45 / tmp46)
    tmp48 = tmp40 * tmp47
    tmp49 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = (tmp48 / tmp49)
    tmp51 = tl.load(in_ptr9 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.where(tmp39, tmp50, tmp51)
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp36, tmp52, tmp53)
    tmp55 = tl.where(tmp19, tmp35, tmp54)
    tmp56 = tl.where(tmp4, tmp15, tmp55)
    tl.store(out_ptr0 + (x1 + 3*y0), tmp56, xmask)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/nj/cnj56jthwnnwojxeih3c3j6sj7ugvom2dtkynk527fnze35cfdtc.py
# Topologically Sorted Source Nodes: [lx, lx_f, ux, ux_f, ly, ly_f, uy, uy_f, getitem_33, reshape_5, As, abc, Bs, Cs, A_2, A_by_ly, isinf, setitem_2, u_1, le_12, l, ge, logical_and, guard, setitem_4, arg, abs_11, le_14, clamp, acos, truediv_39, full_like, where_28, sub_61, cos, mul_86, x1, sub_64, truediv_40, root1, le_15, ge_2, logical_and_2, guard_2, setitem_6, sub_62, cos_1, mul_87, x2, sub_65, truediv_41, root2, le_16, ge_3, logical_and_3, guard_3, setitem_7, sub_63, cos_2, mul_88, x3, sub_66, truediv_42, root3, le_17, ge_4, logical_and_4, guard_4, setitem_8, A_by_uy, isinf_1, setitem_3, le_13, ge_1, logical_and_1, guard_1, setitem_5, arg_1, abs_13, le_18, clamp_1, acos_1, truediv_43, full_like_1, where_29, sub_69, cos_3, mul_94, x1_1, sub_72, truediv_44, root1_uy, le_19, ge_5, logical_and_5, guard_5, setitem_9, sub_70, cos_4, mul_95, x2_1, sub_73, truediv_45, root2_uy, le_20, ge_6, logical_and_6, guard_6, setitem_10, sub_71, cos_5, mul_96, x3_1, sub_74, truediv_46, root3_uy, le_21, ge_7, logical_and_7, guard_7, setitem_11, As_1, mul_101, Bs_1, mul_102, add_51, Cs_1, dual_c, gt_9, lt, or__6, a_nonzero], Original ATen: [aten.sub, aten.view, aten.add, aten.select, aten.clone, aten._unsafe_view, aten.stack, aten.div, aten.isinf, aten.lift_fresh, aten.index_put, aten.neg, aten.le, aten.ge, aten.logical_and, aten.logical_not, aten.mul, aten.abs, aten.clamp, aten.acos, aten.full_like, aten.where, aten.cos, aten.rsub, aten.log, aten.gt, aten.lt, aten.bitwise_or, aten.all]
# Source node to ATen node mapping:
#   A_2 => select_18
#   A_by_ly => div_36
#   A_by_uy => div_37
#   As => add_48
#   As_1 => view_9
#   Bs => select_13
#   Bs_1 => view_10
#   Cs => select_14
#   Cs_1 => view_11
#   a_nonzero => any_1, logical_not
#   abc => cat_2, unsqueeze_20, unsqueeze_21, unsqueeze_22
#   abs_11 => abs_11
#   abs_13 => abs_13
#   acos => acos
#   acos_1 => acos_1
#   add_51 => add_61
#   arg => mul_95
#   arg_1 => mul_103
#   clamp => clamp_max, clamp_min_2
#   clamp_1 => clamp_max_1, clamp_min_3
#   cos => cos
#   cos_1 => cos_1
#   cos_2 => cos_2
#   cos_3 => cos_3
#   cos_4 => cos_4
#   cos_5 => cos_5
#   dual_c => add_62
#   full_like => full_default_28
#   full_like_1 => full_default_35
#   ge => ge
#   ge_1 => ge_1
#   ge_2 => ge_2
#   ge_3 => ge_3
#   ge_4 => ge_4
#   ge_5 => ge_5
#   ge_6 => ge_6
#   ge_7 => ge_7
#   getitem_33 => select_12
#   gt_9 => gt_9
#   guard => logical_not_2
#   guard_1 => logical_not_3
#   guard_2 => logical_not_4
#   guard_3 => logical_not_5
#   guard_4 => logical_not_6
#   guard_5 => logical_not_7
#   guard_6 => logical_not_8
#   guard_7 => logical_not_9
#   isinf => isinf
#   isinf_1 => isinf_1
#   l => full_default_24
#   le_12 => le_12
#   le_13 => le_13
#   le_14 => le_14
#   le_15 => le_15
#   le_16 => le_16
#   le_17 => le_17
#   le_18 => le_18
#   le_19 => le_19
#   le_20 => le_20
#   le_21 => le_21
#   logical_and => logical_and
#   logical_and_1 => logical_and_1
#   logical_and_2 => logical_and_2
#   logical_and_3 => logical_and_3
#   logical_and_4 => logical_and_4
#   logical_and_5 => logical_and_5
#   logical_and_6 => logical_and_6
#   logical_and_7 => logical_and_7
#   lt => lt
#   lx => sub_1
#   lx_f => view
#   ly => sub_3
#   ly_f => view_2
#   mul_101 => mul_111
#   mul_102 => mul_112
#   mul_86 => mul_96
#   mul_87 => mul_97
#   mul_88 => mul_98
#   mul_94 => mul_104
#   mul_95 => mul_105
#   mul_96 => mul_106
#   or__6 => bitwise_or_6
#   reshape_5 => clone_1, view_6
#   root1 => log
#   root1_uy => log_3
#   root2 => log_1
#   root2_uy => log_4
#   root3 => log_2
#   root3_uy => log_5
#   setitem_10 => full_default_37, index_put_8
#   setitem_11 => full_default_38, index_put_9
#   setitem_2 => full_default_22, index_put
#   setitem_3 => full_default_23, index_put_1
#   setitem_4 => full_default_26, index_put_2
#   setitem_5 => full_default_27, index_put_3
#   setitem_6 => full_default_29, index_put_4
#   setitem_7 => full_default_30, index_put_5
#   setitem_8 => full_default_31, index_put_6
#   setitem_9 => full_default_36, index_put_7
#   sub_61 => sub_63
#   sub_62 => sub_64
#   sub_63 => sub_65
#   sub_64 => sub_66
#   sub_65 => sub_67
#   sub_66 => sub_68
#   sub_69 => sub_71
#   sub_70 => sub_72
#   sub_71 => sub_73
#   sub_72 => sub_74
#   sub_73 => sub_75
#   sub_74 => sub_76
#   truediv_39 => div_39
#   truediv_40 => div_40
#   truediv_41 => div_41
#   truediv_42 => div_42
#   truediv_43 => div_43
#   truediv_44 => div_44
#   truediv_45 => div_45
#   truediv_46 => div_46
#   u_1 => full_default_25
#   ux => add_4
#   ux_f => view_1
#   uy => add_9
#   uy_f => view_3
#   where_28 => where_28
#   where_29 => where_32
#   x1 => add_51
#   x1_1 => add_56
#   x2 => add_52
#   x2_1 => add_57
#   x3 => add_53
#   x3_1 => add_58
# Graph fragment:
#   %cat_1 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %arg6_1 : Tensor "f64[32, 1024][0, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %add_48 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_48]
#   %sub_3 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sub_3]
#   %add_9 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=add_9]
#   %index_put : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put]
#   %index_put_1 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put_1]
#   %index_put_2 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put_2]
#   %add_4 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=add_4]
#   %sub_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sub_1]
#   %index_put_3 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put_3]
#   %logical_not_4 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=logical_not_4]
#   %logical_not_5 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=logical_not_5]
#   %logical_not_6 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=logical_not_6]
#   %logical_not_7 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=logical_not_7]
#   %logical_not_8 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=logical_not_8]
#   %logical_not_9 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=logical_not_9]
#   %arg2_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %sub_1 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %view : Tensor "f64[32768][1]cuda:0"[num_users=11] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_1, [-1]), kwargs = {})
#   %add_4 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %view_1 : Tensor "f64[32768][1]cuda:0"[num_users=11] = call_function[target=torch.ops.aten.reshape.default](args = (%add_4, [-1]), kwargs = {})
#   %sub_3 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   %view_2 : Tensor "f64[32768][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_3, [-1]), kwargs = {})
#   %add_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   %view_3 : Tensor "f64[32768][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%add_9, [-1]), kwargs = {})
#   %select_12 : Tensor "f64[32768][3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat_1, 1, 1), kwargs = {})
#   %clone_1 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%arg6_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_1, [32768]), kwargs = {})
#   %add_48 : Tensor "f64[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_12, %view_6), kwargs = {})
#   %unsqueeze_20 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_48, 1), kwargs = {})
#   %select_13 : Tensor "f64[32768][3]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%cat_1, 1, 2), kwargs = {})
#   %unsqueeze_21 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_13, 1), kwargs = {})
#   %select_14 : Tensor "f64[32768][3]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%cat_1, 1, 0), kwargs = {})
#   %unsqueeze_22 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_14, 1), kwargs = {})
#   %cat_2 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_20, %unsqueeze_21, %unsqueeze_22], 1), kwargs = {})
#   %select_18 : Tensor "f64[32768][3]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.select.int](args = (%cat_2, 1, 0), kwargs = {})
#   %div_36 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%select_18, %view_2), kwargs = {})
#   %isinf : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.isinf.default](args = (%div_36,), kwargs = {})
#   %full_default_22 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.index_put_.default](args = (%div_36, [%isinf], %full_default_22), kwargs = {})
#   %full_default_25 : Tensor "f64[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.09622504486493763), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %le_12 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%index_put, %full_default_25), kwargs = {})
#   %full_default_24 : Tensor "f64[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], -0.09622504486493763), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %ge : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%index_put, %full_default_24), kwargs = {})
#   %logical_and : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_12, %ge), kwargs = {})
#   %logical_not_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_26 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put, [%logical_not_2], %full_default_26), kwargs = {})
#   %mul_95 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_put_2, 10.392304845413264), kwargs = {})
#   %abs_11 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_95,), kwargs = {})
#   %le_14 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_11, 1.0000001), kwargs = {})
#   %clamp_min_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_95, -1.0), kwargs = {})
#   %clamp_max : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %acos : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.acos.default](args = (%clamp_max,), kwargs = {})
#   %div_39 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%acos, 3.0), kwargs = {})
#   %full_default_28 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_28 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%le_14, %div_39, %full_default_28), kwargs = {})
#   %sub_63 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_28, 0.0), kwargs = {})
#   %cos : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sub_63,), kwargs = {})
#   %mul_96 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 0.5773502691896258), kwargs = {})
#   %add_51 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, 0.5), kwargs = {})
#   %sub_66 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_51), kwargs = {})
#   %div_40 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_51, %sub_66), kwargs = {})
#   %log : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.log.default](args = (%div_40,), kwargs = {})
#   %le_15 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%log, %view_1), kwargs = {})
#   %ge_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%log, %view), kwargs = {})
#   %logical_and_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_15, %ge_2), kwargs = {})
#   %logical_not_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and_2,), kwargs = {})
#   %full_default_29 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%log, [%logical_not_4], %full_default_29), kwargs = {})
#   %sub_64 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_28, 2.0943951023931953), kwargs = {})
#   %cos_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sub_64,), kwargs = {})
#   %mul_97 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos_1, 0.5773502691896258), kwargs = {})
#   %add_52 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, 0.5), kwargs = {})
#   %sub_67 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_52), kwargs = {})
#   %div_41 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_52, %sub_67), kwargs = {})
#   %log_1 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.log.default](args = (%div_41,), kwargs = {})
#   %le_16 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%log_1, %view_1), kwargs = {})
#   %ge_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%log_1, %view), kwargs = {})
#   %logical_and_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_16, %ge_3), kwargs = {})
#   %logical_not_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and_3,), kwargs = {})
#   %full_default_30 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%log_1, [%logical_not_5], %full_default_30), kwargs = {})
#   %sub_65 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_28, 4.1887902047863905), kwargs = {})
#   %cos_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sub_65,), kwargs = {})
#   %mul_98 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos_2, 0.5773502691896258), kwargs = {})
#   %add_53 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, 0.5), kwargs = {})
#   %sub_68 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_53), kwargs = {})
#   %div_42 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_53, %sub_68), kwargs = {})
#   %log_2 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.log.default](args = (%div_42,), kwargs = {})
#   %le_17 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%log_2, %view_1), kwargs = {})
#   %ge_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%log_2, %view), kwargs = {})
#   %logical_and_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_17, %ge_4), kwargs = {})
#   %logical_not_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and_4,), kwargs = {})
#   %full_default_31 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%log_2, [%logical_not_6], %full_default_31), kwargs = {})
#   %div_37 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%select_18, %view_3), kwargs = {})
#   %isinf_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.isinf.default](args = (%div_37,), kwargs = {})
#   %full_default_23 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_1 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.index_put_.default](args = (%div_37, [%isinf_1], %full_default_23), kwargs = {})
#   %le_13 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%index_put_1, %full_default_25), kwargs = {})
#   %ge_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%index_put_1, %full_default_24), kwargs = {})
#   %logical_and_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_13, %ge_1), kwargs = {})
#   %logical_not_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and_1,), kwargs = {})
#   %full_default_27 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_1, [%logical_not_3], %full_default_27), kwargs = {})
#   %mul_103 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index_put_3, 10.392304845413264), kwargs = {})
#   %abs_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_103,), kwargs = {})
#   %le_18 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%abs_13, 1.0000001), kwargs = {})
#   %clamp_min_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_103, -1.0), kwargs = {})
#   %clamp_max_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %acos_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.acos.default](args = (%clamp_max_1,), kwargs = {})
#   %div_43 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%acos_1, 3.0), kwargs = {})
#   %full_default_35 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_32 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%le_18, %div_43, %full_default_35), kwargs = {})
#   %sub_71 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_32, 0.0), kwargs = {})
#   %cos_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sub_71,), kwargs = {})
#   %mul_104 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos_3, 0.5773502691896258), kwargs = {})
#   %add_56 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, 0.5), kwargs = {})
#   %sub_74 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_56), kwargs = {})
#   %div_44 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_56, %sub_74), kwargs = {})
#   %log_3 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.log.default](args = (%div_44,), kwargs = {})
#   %le_19 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%log_3, %view_1), kwargs = {})
#   %ge_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%log_3, %view), kwargs = {})
#   %logical_and_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_19, %ge_5), kwargs = {})
#   %logical_not_7 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and_5,), kwargs = {})
#   %full_default_36 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%log_3, [%logical_not_7], %full_default_36), kwargs = {})
#   %sub_72 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_32, 2.0943951023931953), kwargs = {})
#   %cos_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sub_72,), kwargs = {})
#   %mul_105 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos_4, 0.5773502691896258), kwargs = {})
#   %add_57 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, 0.5), kwargs = {})
#   %sub_75 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_57), kwargs = {})
#   %div_45 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_57, %sub_75), kwargs = {})
#   %log_4 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.log.default](args = (%div_45,), kwargs = {})
#   %le_20 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%log_4, %view_1), kwargs = {})
#   %ge_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%log_4, %view), kwargs = {})
#   %logical_and_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_20, %ge_6), kwargs = {})
#   %logical_not_8 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and_6,), kwargs = {})
#   %full_default_37 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%log_4, [%logical_not_8], %full_default_37), kwargs = {})
#   %sub_73 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_32, 4.1887902047863905), kwargs = {})
#   %cos_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sub_73,), kwargs = {})
#   %mul_106 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos_5, 0.5773502691896258), kwargs = {})
#   %add_58 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, 0.5), kwargs = {})
#   %sub_76 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_58), kwargs = {})
#   %div_46 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_58, %sub_76), kwargs = {})
#   %log_5 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.log.default](args = (%div_46,), kwargs = {})
#   %le_21 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%log_5, %view_1), kwargs = {})
#   %ge_7 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%log_5, %view), kwargs = {})
#   %logical_and_7 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_21, %ge_7), kwargs = {})
#   %logical_not_9 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%logical_and_7,), kwargs = {})
#   %full_default_38 : Tensor "f64[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], nan), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%log_5, [%logical_not_9], %full_default_38), kwargs = {})
#   %view_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_48, [32, 1024]), kwargs = {})
#   %mul_111 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %arg2_1), kwargs = {})
#   %view_10 : Tensor "f64[32, 1024][3072, 3]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%select_13, [32, 1024]), kwargs = {})
#   %mul_112 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_10, %arg3_1), kwargs = {})
#   %add_61 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_111, %mul_112), kwargs = {})
#   %view_11 : Tensor "f64[32, 1024][3072, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_14, [32, 1024]), kwargs = {})
#   %add_62 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_61, %view_11), kwargs = {})
#   %gt_9 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_48, 0), kwargs = {})
#   %lt : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_48, 0), kwargs = {})
#   %bitwise_or_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%gt_9, %lt), kwargs = {})
#   %logical_not : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%bitwise_or_6,), kwargs = {})
#   %any_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.any.dims](args = (%logical_not,), kwargs = {})
#   return %add_48,%index_put,%index_put_1,%index_put_2,%index_put_3,%logical_not_4,%logical_not_5,%logical_not_6,%logical_not_7,%logical_not_8,%logical_not_9,%index_put_4,%index_put_5,%index_put_6,%index_put_7,%index_put_8,%index_put_9,%add_62,%buf85
triton_red_fused__unsafe_view_abs_acos_add_all_bitwise_or_clamp_clone_cos_div_full_like_ge_gt_index_put_isinf_le_lift_fresh_log_logical_and_logical_not_lt_mul_neg_rsub_select_stack_sub_view_where_6 = async_compile.triton('triton_red_fused__unsafe_view_abs_acos_add_all_bitwise_or_clamp_clone_cos_div_full_like_ge_gt_index_put_isinf_le_lift_fresh_log_logical_and_logical_not_lt_mul_neg_rsub_select_stack_sub_view_where_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'in_ptr4': '*fp64', 'in_ptr5': '*fp64', 'in_ptr6': '*fp64', 'in_ptr7': '*fp64', 'out_ptr0': '*fp64', 'out_ptr7': '*fp64', 'out_ptr8': '*fp64', 'out_ptr9': '*fp64', 'out_ptr10': '*fp64', 'out_ptr11': '*fp64', 'out_ptr12': '*fp64', 'out_ptr13': '*fp64', 'out_ptr14': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_abs_acos_add_all_bitwise_or_clamp_clone_cos_div_full_like_ge_gt_index_put_isinf_le_lift_fresh_log_logical_and_logical_not_lt_mul_neg_rsub_select_stack_sub_view_where_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 9, 'num_reduction': 1, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4, 'r0_': 6094848}}
)
@triton.jit
def triton_red_fused__unsafe_view_abs_acos_add_all_bitwise_or_clamp_clone_cos_div_full_like_ge_gt_index_put_isinf_le_lift_fresh_log_logical_and_logical_not_lt_mul_neg_rsub_select_stack_sub_view_where_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp152 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (1 + 3*r0_1 + 24576*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((r0_1 % 1024)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr3 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp62 = tl.load(in_ptr4 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp64 = tl.load(in_ptr5 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp139 = tl.load(in_ptr6 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp141 = tl.load(in_ptr0 + (2 + 3*r0_1 + 24576*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp142 = tl.load(in_ptr7 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp145 = tl.load(in_ptr0 + (3*r0_1 + 24576*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tmp3 >= tmp3
        tmp5 = tl.full([1, 1], 1, tl.int64)
        tmp6 = tmp3 < tmp5
        tmp7 = tmp3 >= tmp5
        tmp8 = tl.full([1, 1], 2, tl.int64)
        tmp9 = tmp3 < tmp8
        tmp10 = tmp7 & tmp9
        tmp11 = tl.load(in_ptr0 + (2 + 3*r0_1 + 24576*x0), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp3 >= tmp8
        tmp13 = tl.full([1, 1], 3, tl.int64)
        tmp14 = tmp3 < tmp13
        tmp15 = tl.load(in_ptr0 + (3*r0_1 + 24576*x0), r0_mask & tmp12 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.where(tmp10, tmp11, tmp15)
        tmp17 = tl.where(tmp6, tmp2, tmp16)
        tmp19 = (tmp17 / tmp18)
        tmp20 = libdevice.isinf(tmp19).to(tl.int1)
        tmp21 = tl.full([1, 1], float("nan"), tl.float64)
        tmp22 = tl.where(tmp20, tmp21, tmp19)
        tmp24 = (tmp17 / tmp23)
        tmp25 = libdevice.isinf(tmp24).to(tl.int1)
        tmp26 = tl.where(tmp25, tmp21, tmp24)
        tmp27 = tl.full([1, 1], 0.09622504486493763, tl.float64)
        tmp28 = tmp22 <= tmp27
        tmp29 = tl.full([1, 1], -0.09622504486493763, tl.float64)
        tmp30 = tmp22 >= tmp29
        tmp31 = tmp28 & tmp30
        tmp32 = tmp31 == 0
        tmp33 = tl.where(tmp32, tmp21, tmp22)
        tmp34 = tmp26 <= tmp27
        tmp35 = tmp26 >= tmp29
        tmp36 = tmp34 & tmp35
        tmp37 = tmp36 == 0
        tmp38 = tl.where(tmp37, tmp21, tmp26)
        tmp39 = tl.full([1, 1], 10.392304845413264, tl.float64)
        tmp40 = tmp33 * tmp39
        tmp41 = tl_math.abs(tmp40)
        tmp42 = tl.full([1, 1], 1.0000001, tl.float64)
        tmp43 = tmp41 <= tmp42
        tmp44 = tl.full([1, 1], -1.0, tl.float64)
        tmp45 = triton_helpers.maximum(tmp40, tmp44)
        tmp46 = tl.full([1, 1], 1.0, tl.float64)
        tmp47 = triton_helpers.minimum(tmp45, tmp46)
        tmp48 = libdevice.acos(tmp47)
        tmp49 = tl.full([1, 1], 0.3333333333333333, tl.float64)
        tmp50 = tmp48 * tmp49
        tmp51 = tl.where(tmp43, tmp50, tmp21)
        tmp52 = tl.full([1, 1], 0.0, tl.float64)
        tmp53 = tmp51 - tmp52
        tmp54 = libdevice.cos(tmp53)
        tmp55 = tl.full([1, 1], 0.5773502691896258, tl.float64)
        tmp56 = tmp54 * tmp55
        tmp57 = tl.full([1, 1], 0.5, tl.float64)
        tmp58 = tmp56 + tmp57
        tmp59 = tmp46 - tmp58
        tmp60 = (tmp58 / tmp59)
        tmp61 = libdevice.log(tmp60)
        tmp63 = tmp61 <= tmp62
        tmp65 = tmp61 >= tmp64
        tmp66 = tmp63 & tmp65
        tmp67 = tmp66 == 0
        tmp68 = tl.full([1, 1], 2.0943951023931953, tl.float64)
        tmp69 = tmp51 - tmp68
        tmp70 = libdevice.cos(tmp69)
        tmp71 = tmp70 * tmp55
        tmp72 = tmp71 + tmp57
        tmp73 = tmp46 - tmp72
        tmp74 = (tmp72 / tmp73)
        tmp75 = libdevice.log(tmp74)
        tmp76 = tmp75 <= tmp62
        tmp77 = tmp75 >= tmp64
        tmp78 = tmp76 & tmp77
        tmp79 = tmp78 == 0
        tmp80 = tl.full([1, 1], 4.1887902047863905, tl.float64)
        tmp81 = tmp51 - tmp80
        tmp82 = libdevice.cos(tmp81)
        tmp83 = tmp82 * tmp55
        tmp84 = tmp83 + tmp57
        tmp85 = tmp46 - tmp84
        tmp86 = (tmp84 / tmp85)
        tmp87 = libdevice.log(tmp86)
        tmp88 = tmp87 <= tmp62
        tmp89 = tmp87 >= tmp64
        tmp90 = tmp88 & tmp89
        tmp91 = tmp90 == 0
        tmp92 = tmp38 * tmp39
        tmp93 = tl_math.abs(tmp92)
        tmp94 = tmp93 <= tmp42
        tmp95 = triton_helpers.maximum(tmp92, tmp44)
        tmp96 = triton_helpers.minimum(tmp95, tmp46)
        tmp97 = libdevice.acos(tmp96)
        tmp98 = tmp97 * tmp49
        tmp99 = tl.where(tmp94, tmp98, tmp21)
        tmp100 = tmp99 - tmp52
        tmp101 = libdevice.cos(tmp100)
        tmp102 = tmp101 * tmp55
        tmp103 = tmp102 + tmp57
        tmp104 = tmp46 - tmp103
        tmp105 = (tmp103 / tmp104)
        tmp106 = libdevice.log(tmp105)
        tmp107 = tmp106 <= tmp62
        tmp108 = tmp106 >= tmp64
        tmp109 = tmp107 & tmp108
        tmp110 = tmp109 == 0
        tmp111 = tmp99 - tmp68
        tmp112 = libdevice.cos(tmp111)
        tmp113 = tmp112 * tmp55
        tmp114 = tmp113 + tmp57
        tmp115 = tmp46 - tmp114
        tmp116 = (tmp114 / tmp115)
        tmp117 = libdevice.log(tmp116)
        tmp118 = tmp117 <= tmp62
        tmp119 = tmp117 >= tmp64
        tmp120 = tmp118 & tmp119
        tmp121 = tmp120 == 0
        tmp122 = tmp99 - tmp80
        tmp123 = libdevice.cos(tmp122)
        tmp124 = tmp123 * tmp55
        tmp125 = tmp124 + tmp57
        tmp126 = tmp46 - tmp125
        tmp127 = (tmp125 / tmp126)
        tmp128 = libdevice.log(tmp127)
        tmp129 = tmp128 <= tmp62
        tmp130 = tmp128 >= tmp64
        tmp131 = tmp129 & tmp130
        tmp132 = tmp131 == 0
        tmp133 = tl.where(tmp67, tmp21, tmp61)
        tmp134 = tl.where(tmp79, tmp21, tmp75)
        tmp135 = tl.where(tmp91, tmp21, tmp87)
        tmp136 = tl.where(tmp110, tmp21, tmp106)
        tmp137 = tl.where(tmp121, tmp21, tmp117)
        tmp138 = tl.where(tmp132, tmp21, tmp128)
        tmp140 = tmp2 * tmp139
        tmp143 = tmp141 * tmp142
        tmp144 = tmp140 + tmp143
        tmp146 = tmp144 + tmp145
        tmp147 = tmp2 > tmp52
        tmp148 = tmp2 < tmp52
        tmp149 = tmp147 | tmp148
        tmp150 = tmp149 == 0
        tmp151 = tl.broadcast_to(tmp150, [XBLOCK, R0_BLOCK])
        tmp153 = _tmp152 | tmp151
        _tmp152 = tl.where(r0_mask & xmask, tmp153, _tmp152)
        tl.store(out_ptr0 + (r0_1 + 8192*x0), tmp2, r0_mask & xmask)
        tl.store(out_ptr7 + (r0_1 + 8192*x0), tmp133, r0_mask & xmask)
        tl.store(out_ptr8 + (r0_1 + 8192*x0), tmp134, r0_mask & xmask)
        tl.store(out_ptr9 + (r0_1 + 8192*x0), tmp135, r0_mask & xmask)
        tl.store(out_ptr10 + (r0_1 + 8192*x0), tmp136, r0_mask & xmask)
        tl.store(out_ptr11 + (r0_1 + 8192*x0), tmp137, r0_mask & xmask)
        tl.store(out_ptr12 + (r0_1 + 8192*x0), tmp138, r0_mask & xmask)
        tl.store(out_ptr13 + (r0_1 + 8192*x0), tmp146, r0_mask & xmask)
    tmp154 = _tmp152.to(tl.int8)
    tmp152 = triton_helpers.any(tmp154, 1)[:, None]
    tl.store(out_ptr14 + (x0), tmp152, xmask)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/p5/cp5kv3rqmri5x3trafa74btrool7sy7zuvyb76r6q7zpb3bcfgjq.py
# Topologically Sorted Source Nodes: [gt_9, lt, or__6, a_nonzero], Original ATen: [aten.gt, aten.lt, aten.bitwise_or, aten.all]
# Source node to ATen node mapping:
#   a_nonzero => any_1, logical_not, logical_not_1
#   gt_9 => gt_9
#   lt => lt
#   or__6 => bitwise_or_6
# Graph fragment:
#   %buf85 : Tensor "b8[4][1]cuda:0" = PlaceHolder[target=buf85]
#   %any_1 : Tensor "b8[][]cuda:0" = PlaceHolder[target=any_1]
#   %gt_9 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_48, 0), kwargs = {})
#   %lt : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%add_48, 0), kwargs = {})
#   %bitwise_or_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%gt_9, %lt), kwargs = {})
#   %logical_not : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%bitwise_or_6,), kwargs = {})
#   %any_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.any.dims](args = (%logical_not,), kwargs = {})
#   %logical_not_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   return %any_1,%logical_not_1
triton_per_fused_all_bitwise_or_gt_lt_7 = async_compile.triton('triton_per_fused_all_bitwise_or_gt_lt_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*i1', 'in_ptr0': '*i1', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_all_bitwise_or_gt_lt_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 2}}
)
@triton.jit
def triton_per_fused_all_bitwise_or_gt_lt_7(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None).to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = triton_helpers.any(tmp1, 1)[:, None].to(tl.int1)
    tmp4 = tmp3 == 0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp4, None)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/vq/cvqr7jdfa73n6gyoklrbypizbk7k457yjcdheb4ndrdabvr2tgfg.py
# Topologically Sorted Source Nodes: [ly, ly_f, uy, uy_f, abc, Bs, Cs, A_2, stack_5, sig_1, sub_67, roots_y, mul_89, sigmoid_prime_y_1, unsqueeze_12, mul_91, B_2, unsqueeze_13, mul_92, add_44, C_2, unsqueeze_14, planar_1, sub_68, evaluation_1, stack_6, sig_2, sub_75, roots_y_uy, mul_97, sigmoid_prime_y_2, unsqueeze_16, mul_99, unsqueeze_17, mul_100, add_49, unsqueeze_18, planar_2, sub_76, evaluation_3], Original ATen: [aten.sub, aten.view, aten.add, aten.stack, aten.select, aten.sigmoid, aten.rsub, aten.unsqueeze, aten.mul, aten.abs]
# Source node to ATen node mapping:
#   A_2 => select_18
#   B_2 => select_19
#   Bs => select_13
#   C_2 => select_20
#   Cs => select_14
#   abc => cat_2, unsqueeze_20, unsqueeze_21, unsqueeze_22
#   add_44 => add_54
#   add_49 => add_59
#   evaluation_1 => abs_12
#   evaluation_3 => abs_14
#   ly => sub_3
#   ly_f => view_2
#   mul_100 => mul_110
#   mul_89 => mul_99
#   mul_91 => mul_101
#   mul_92 => mul_102
#   mul_97 => mul_107
#   mul_99 => mul_109
#   planar_1 => add_55
#   planar_2 => add_60
#   roots_y => unsqueeze_37
#   roots_y_uy => unsqueeze_44
#   sig_1 => sigmoid_5
#   sig_2 => sigmoid_6
#   sigmoid_prime_y_1 => mul_100
#   sigmoid_prime_y_2 => mul_108
#   stack_5 => cat_5, unsqueeze_34, unsqueeze_35, unsqueeze_36
#   stack_6 => cat_6, unsqueeze_41, unsqueeze_42, unsqueeze_43
#   sub_67 => sub_69
#   sub_68 => sub_70
#   sub_75 => sub_77
#   sub_76 => sub_78
#   unsqueeze_12 => unsqueeze_38
#   unsqueeze_13 => unsqueeze_39
#   unsqueeze_14 => unsqueeze_40
#   unsqueeze_16 => unsqueeze_45
#   unsqueeze_17 => unsqueeze_46
#   unsqueeze_18 => unsqueeze_47
#   uy => add_9
#   uy_f => view_3
# Graph fragment:
#   %add_48 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_48]
#   %cat_1 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %index_put_4 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put_4]
#   %index_put_5 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put_5]
#   %index_put_6 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put_6]
#   %mul_101 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=mul_101]
#   %sub_3 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sub_3]
#   %add_55 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=add_55]
#   %index_put_7 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put_7]
#   %index_put_8 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put_8]
#   %index_put_9 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=index_put_9]
#   %mul_109 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=mul_109]
#   %add_9 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=add_9]
#   %add_60 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=add_60]
#   %sub_3 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   %view_2 : Tensor "f64[32768][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_3, [-1]), kwargs = {})
#   %add_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   %view_3 : Tensor "f64[32768][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%add_9, [-1]), kwargs = {})
#   %unsqueeze_20 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_48, 1), kwargs = {})
#   %select_13 : Tensor "f64[32768][3]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%cat_1, 1, 2), kwargs = {})
#   %unsqueeze_21 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_13, 1), kwargs = {})
#   %select_14 : Tensor "f64[32768][3]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%cat_1, 1, 0), kwargs = {})
#   %unsqueeze_22 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_14, 1), kwargs = {})
#   %cat_2 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_20, %unsqueeze_21, %unsqueeze_22], 1), kwargs = {})
#   %select_18 : Tensor "f64[32768][3]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.select.int](args = (%cat_2, 1, 0), kwargs = {})
#   %unsqueeze_34 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put_4, 1), kwargs = {})
#   %unsqueeze_35 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put_5, 1), kwargs = {})
#   %unsqueeze_36 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put_6, 1), kwargs = {})
#   %cat_5 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_34, %unsqueeze_35, %unsqueeze_36], 1), kwargs = {})
#   %sigmoid_5 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%cat_5,), kwargs = {})
#   %sub_69 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid_5), kwargs = {})
#   %unsqueeze_37 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 1), kwargs = {})
#   %mul_99 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_5, %unsqueeze_37), kwargs = {})
#   %mul_100 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %mul_99), kwargs = {})
#   %unsqueeze_38 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_18, 1), kwargs = {})
#   %mul_101 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_38, %cat_5), kwargs = {})
#   %select_19 : Tensor "f64[32768][3]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%cat_2, 1, 1), kwargs = {})
#   %unsqueeze_39 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_19, 1), kwargs = {})
#   %mul_102 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_39, %unsqueeze_37), kwargs = {})
#   %add_54 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %mul_102), kwargs = {})
#   %select_20 : Tensor "f64[32768][3]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%cat_2, 1, 2), kwargs = {})
#   %unsqueeze_40 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_20, 1), kwargs = {})
#   %add_55 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %unsqueeze_40), kwargs = {})
#   %sub_70 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_100, %add_55), kwargs = {})
#   %abs_12 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.abs.default](args = (%sub_70,), kwargs = {})
#   %unsqueeze_41 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put_7, 1), kwargs = {})
#   %unsqueeze_42 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put_8, 1), kwargs = {})
#   %unsqueeze_43 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put_9, 1), kwargs = {})
#   %cat_6 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_41, %unsqueeze_42, %unsqueeze_43], 1), kwargs = {})
#   %sigmoid_6 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%cat_6,), kwargs = {})
#   %sub_77 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid_6), kwargs = {})
#   %unsqueeze_44 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 1), kwargs = {})
#   %mul_107 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_6, %unsqueeze_44), kwargs = {})
#   %mul_108 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %mul_107), kwargs = {})
#   %unsqueeze_45 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_18, 1), kwargs = {})
#   %mul_109 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_45, %cat_6), kwargs = {})
#   %unsqueeze_46 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_19, 1), kwargs = {})
#   %mul_110 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_46, %unsqueeze_44), kwargs = {})
#   %add_59 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_109, %mul_110), kwargs = {})
#   %unsqueeze_47 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_20, 1), kwargs = {})
#   %add_60 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_59, %unsqueeze_47), kwargs = {})
#   %sub_78 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_108, %add_60), kwargs = {})
#   %abs_14 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.abs.default](args = (%sub_78,), kwargs = {})
#   return %mul_101,%add_55,%abs_12,%mul_109,%add_60,%abs_14
triton_poi_fused_abs_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_8 = async_compile.triton('triton_poi_fused_abs_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_out_ptr1': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'in_ptr4': '*fp64', 'in_ptr5': '*fp64', 'in_ptr6': '*fp64', 'in_ptr7': '*fp64', 'in_ptr8': '*fp64', 'in_ptr9': '*fp64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 17, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp42 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp0 >= tmp2
    tmp6 = tl.full([1, 1], 2, tl.int64)
    tmp7 = tmp0 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tl.load(in_ptr1 + (tl.broadcast_to(2 + 3*y0, [YBLOCK, XBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp0 >= tmp6
    tmp11 = tl.full([1, 1], 3, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr1 + (tl.broadcast_to(3*y0, [YBLOCK, XBLOCK])), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp8, tmp9, tmp13)
    tmp15 = tl.where(tmp3, tmp4, tmp14)
    tmp16 = x1
    tmp17 = tmp16 >= tmp0
    tmp18 = tmp16 < tmp2
    tmp19 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp16 >= tmp2
    tmp21 = tmp16 < tmp6
    tmp22 = tmp20 & tmp21
    tmp23 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp16 >= tmp6
    tmp25 = tmp16 < tmp11
    tmp26 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp22, tmp23, tmp26)
    tmp28 = tl.where(tmp18, tmp19, tmp27)
    tmp29 = tmp15 * tmp28
    tmp30 = tmp2 >= tmp0
    tmp31 = tmp2 < tmp2
    tmp32 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp2 >= tmp2
    tmp34 = tmp2 < tmp6
    tmp35 = tmp33 & tmp34
    tmp36 = tl.load(in_ptr1 + (tl.broadcast_to(2 + 3*y0, [YBLOCK, XBLOCK])), tmp35 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp2 >= tmp6
    tmp38 = tmp2 < tmp11
    tmp39 = tl.load(in_ptr1 + (tl.broadcast_to(3*y0, [YBLOCK, XBLOCK])), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp35, tmp36, tmp39)
    tmp41 = tl.where(tmp31, tmp32, tmp40)
    tmp43 = tmp41 * tmp42
    tmp44 = tmp29 + tmp43
    tmp45 = tmp6 >= tmp0
    tmp46 = tmp6 < tmp2
    tmp47 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp6 >= tmp2
    tmp49 = tmp6 < tmp6
    tmp50 = tmp48 & tmp49
    tmp51 = tl.load(in_ptr1 + (tl.broadcast_to(2 + 3*y0, [YBLOCK, XBLOCK])), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp6 >= tmp6
    tmp53 = tmp6 < tmp11
    tmp54 = tl.load(in_ptr1 + (tl.broadcast_to(3*y0, [YBLOCK, XBLOCK])), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tl.where(tmp50, tmp51, tmp54)
    tmp56 = tl.where(tmp46, tmp47, tmp55)
    tmp57 = tmp44 + tmp56
    tmp58 = tl.full([1, 1], 1, tl.int32)
    tmp59 = -tmp28
    tmp60 = libdevice.exp(tmp59)
    tmp61 = tmp58 + tmp60
    tmp62 = (tmp58 / tmp61)
    tmp63 = tl.full([1, 1], 1.0, tl.float64)
    tmp64 = tmp63 - tmp62
    tmp65 = tmp62 * tmp42
    tmp66 = tmp64 * tmp65
    tmp67 = tmp66 - tmp57
    tmp68 = tl_math.abs(tmp67)
    tmp69 = tl.load(in_ptr6 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tl.load(in_ptr7 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.load(in_ptr8 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp72 = tl.where(tmp22, tmp70, tmp71)
    tmp73 = tl.where(tmp18, tmp69, tmp72)
    tmp74 = tmp15 * tmp73
    tmp76 = tmp41 * tmp75
    tmp77 = tmp74 + tmp76
    tmp78 = tmp77 + tmp56
    tmp79 = -tmp73
    tmp80 = libdevice.exp(tmp79)
    tmp81 = tmp58 + tmp80
    tmp82 = (tmp58 / tmp81)
    tmp83 = tmp63 - tmp82
    tmp84 = tmp82 * tmp75
    tmp85 = tmp83 * tmp84
    tmp86 = tmp85 - tmp78
    tmp87 = tl_math.abs(tmp86)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 3*y0), tmp68, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x1 + 3*y0), tmp87, xmask)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/jb/cjbmeehttaicog4stcsbus2yhfaealtfbt6qpttoon7agnhhfvwc.py
# Topologically Sorted Source Nodes: [lx, lx_f, ux, ux_f, ly, ly_f, uy, uy_f, xs_1, sig, sub_59, ys_1, mul_81, sigmoid_prime_y, abc, Bs, Cs, A_1, unsqueeze_8, mul_83, B_1, unsqueeze_9, mul_84, add_39, C_1, unsqueeze_10, planar, evaluation], Original ATen: [aten.sub, aten.view, aten.add, aten.stack, aten.sigmoid, aten.rsub, aten.mul, aten.select, aten.unsqueeze]
# Source node to ATen node mapping:
#   A_1 => select_15
#   B_1 => select_16
#   Bs => select_13
#   C_1 => select_17
#   Cs => select_14
#   abc => cat_2, unsqueeze_20, unsqueeze_21, unsqueeze_22
#   add_39 => add_49
#   evaluation => sub_62
#   lx => sub_1
#   lx_f => view
#   ly => sub_3
#   ly_f => view_2
#   mul_81 => mul_90
#   mul_83 => mul_92
#   mul_84 => mul_93
#   planar => add_50
#   sig => sigmoid_4
#   sigmoid_prime_y => mul_91
#   sub_59 => sub_61
#   unsqueeze_10 => unsqueeze_33
#   unsqueeze_8 => unsqueeze_31
#   unsqueeze_9 => unsqueeze_32
#   ux => add_4
#   ux_f => view_1
#   uy => add_9
#   uy_f => view_3
#   xs_1 => cat_3, unsqueeze_23, unsqueeze_24, unsqueeze_25, unsqueeze_26
#   ys_1 => cat_4, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30
# Graph fragment:
#   %sub_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sub_1]
#   %add_4 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=add_4]
#   %sub_3 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sub_3]
#   %add_9 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=add_9]
#   %add_48 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_48]
#   %cat_1 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %mul_90 : Tensor "f64[32768, 4][4, 1]cuda:0" = PlaceHolder[target=mul_90]
#   %mul_92 : Tensor "f64[32768, 4][4, 1]cuda:0" = PlaceHolder[target=mul_92]
#   %mul_93 : Tensor "f64[32768, 4][4, 1]cuda:0" = PlaceHolder[target=mul_93]
#   %sub_1 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %view : Tensor "f64[32768][1]cuda:0"[num_users=11] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_1, [-1]), kwargs = {})
#   %add_4 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %sum_1), kwargs = {})
#   %view_1 : Tensor "f64[32768][1]cuda:0"[num_users=11] = call_function[target=torch.ops.aten.reshape.default](args = (%add_4, [-1]), kwargs = {})
#   %sub_3 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   %view_2 : Tensor "f64[32768][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_3, [-1]), kwargs = {})
#   %add_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %sum_2), kwargs = {})
#   %view_3 : Tensor "f64[32768][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.reshape.default](args = (%add_9, [-1]), kwargs = {})
#   %unsqueeze_23 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view, 1), kwargs = {})
#   %unsqueeze_24 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view, 1), kwargs = {})
#   %unsqueeze_25 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_1, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_1, 1), kwargs = {})
#   %cat_3 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_23, %unsqueeze_24, %unsqueeze_25, %unsqueeze_26], 1), kwargs = {})
#   %sigmoid_4 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%cat_3,), kwargs = {})
#   %sub_61 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid_4), kwargs = {})
#   %unsqueeze_27 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 1), kwargs = {})
#   %unsqueeze_28 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 1), kwargs = {})
#   %unsqueeze_29 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 1), kwargs = {})
#   %unsqueeze_30 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_3, 1), kwargs = {})
#   %cat_4 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_27, %unsqueeze_28, %unsqueeze_29, %unsqueeze_30], 1), kwargs = {})
#   %mul_90 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_4, %cat_4), kwargs = {})
#   %mul_91 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %mul_90), kwargs = {})
#   %unsqueeze_20 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_48, 1), kwargs = {})
#   %select_13 : Tensor "f64[32768][3]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%cat_1, 1, 2), kwargs = {})
#   %unsqueeze_21 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_13, 1), kwargs = {})
#   %select_14 : Tensor "f64[32768][3]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%cat_1, 1, 0), kwargs = {})
#   %unsqueeze_22 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_14, 1), kwargs = {})
#   %cat_2 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_20, %unsqueeze_21, %unsqueeze_22], 1), kwargs = {})
#   %select_15 : Tensor "f64[32768][3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat_2, 1, 0), kwargs = {})
#   %unsqueeze_31 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_15, 1), kwargs = {})
#   %mul_92 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %cat_3), kwargs = {})
#   %select_16 : Tensor "f64[32768][3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat_2, 1, 1), kwargs = {})
#   %unsqueeze_32 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_16, 1), kwargs = {})
#   %mul_93 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_32, %cat_4), kwargs = {})
#   %add_49 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %mul_93), kwargs = {})
#   %select_17 : Tensor "f64[32768][3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat_2, 1, 2), kwargs = {})
#   %unsqueeze_33 : Tensor "f64[32768, 1][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_17, 1), kwargs = {})
#   %add_50 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %unsqueeze_33), kwargs = {})
#   %sub_62 : Tensor "f64[32768, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_91, %add_50), kwargs = {})
#   return %mul_90,%mul_93,%mul_92,%sub_62
triton_poi_fused_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_9 = async_compile.triton('triton_poi_fused_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'in_ptr4': '*fp64', 'in_ptr5': '*fp64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 17, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1310720, 'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1, 1], 3, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1, 1], 4, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp23 = tl.full([1, 1], 1, tl.int32)
    tmp24 = -tmp22
    tmp25 = libdevice.exp(tmp24)
    tmp26 = tmp23 + tmp25
    tmp27 = (tmp23 / tmp26)
    tmp28 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.where(tmp14, tmp30, tmp31)
    tmp33 = tl.where(tmp9, tmp29, tmp32)
    tmp34 = tl.where(tmp4, tmp28, tmp33)
    tmp35 = tmp27 * tmp34
    tmp36 = tmp3 >= tmp1
    tmp37 = tmp3 < tmp3
    tmp38 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp3 >= tmp3
    tmp40 = tmp3 < tmp7
    tmp41 = tmp39 & tmp40
    tmp42 = tl.load(in_ptr5 + (tl.broadcast_to(2 + 3*y0, [YBLOCK, XBLOCK])), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp3 >= tmp7
    tmp44 = tmp3 < tmp12
    tmp45 = tl.load(in_ptr5 + (tl.broadcast_to(3*y0, [YBLOCK, XBLOCK])), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.where(tmp41, tmp42, tmp45)
    tmp47 = tl.where(tmp37, tmp38, tmp46)
    tmp48 = tmp47 * tmp34
    tmp49 = tmp1 >= tmp1
    tmp50 = tmp1 < tmp3
    tmp51 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp1 >= tmp3
    tmp53 = tmp1 < tmp7
    tmp54 = tmp52 & tmp53
    tmp55 = tl.load(in_ptr5 + (tl.broadcast_to(2 + 3*y0, [YBLOCK, XBLOCK])), tmp54 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp1 >= tmp7
    tmp57 = tmp1 < tmp12
    tmp58 = tl.load(in_ptr5 + (tl.broadcast_to(3*y0, [YBLOCK, XBLOCK])), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tl.where(tmp54, tmp55, tmp58)
    tmp60 = tl.where(tmp50, tmp51, tmp59)
    tmp61 = tmp60 * tmp22
    tmp62 = tl.full([1, 1], 1.0, tl.float64)
    tmp63 = tmp62 - tmp27
    tmp64 = tmp63 * tmp35
    tmp65 = tmp61 + tmp48
    tmp66 = tmp7 >= tmp1
    tmp67 = tmp7 < tmp3
    tmp68 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp7 >= tmp3
    tmp70 = tmp7 < tmp7
    tmp71 = tmp69 & tmp70
    tmp72 = tl.load(in_ptr5 + (tl.broadcast_to(2 + 3*y0, [YBLOCK, XBLOCK])), tmp71 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tmp7 >= tmp7
    tmp74 = tmp7 < tmp12
    tmp75 = tl.load(in_ptr5 + (tl.broadcast_to(3*y0, [YBLOCK, XBLOCK])), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.where(tmp71, tmp72, tmp75)
    tmp77 = tl.where(tmp67, tmp68, tmp76)
    tmp78 = tmp65 + tmp77
    tmp79 = tmp64 - tmp78
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 4*y0), tmp79, xmask)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/cw/ccwgsrwqpl257bf7bzvlhtgptkiy6rxoupod5enzuee4inbawb3z.py
# Topologically Sorted Source Nodes: [Bs, s_lb, sub_2, mul, s_ub, sub_3, mul_1, lam, new_rg, setitem_12, unsqueeze_19, mul_103, As_1, Bs_1, new_dg, setitem_13, unsqueeze_20, mul_104, unsqueeze_21, mul_105, add_56], Original ATen: [aten.select, aten.sigmoid, aten.rsub, aten.mul, aten.minimum, aten.new_zeros, aten.slice, aten.unsqueeze, aten.copy, aten.view, aten.add]
# Source node to ATen node mapping:
#   As_1 => view_9
#   Bs => select_13
#   Bs_1 => view_10
#   add_56 => add_114
#   lam => minimum
#   mul => mul_6
#   mul_1 => mul_7
#   mul_103 => mul_116
#   mul_104 => mul_151
#   mul_105 => mul_155
#   new_dg => full_25
#   new_rg => full_24
#   s_lb => sigmoid
#   s_ub => sigmoid_1
#   setitem_12 => copy_2, slice_13
#   setitem_13 => copy_4, slice_20
#   sub_2 => sub_4
#   sub_3 => sub_5
#   unsqueeze_19 => unsqueeze_48
#   unsqueeze_20 => unsqueeze_49
#   unsqueeze_21 => unsqueeze_50
# Graph fragment:
#   %sub_1 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=sub_1]
#   %add_4 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=add_4]
#   %arg1_1 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add_48 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_48]
#   %cat_1 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %arg5_1 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %select_13 : Tensor "f64[32768][3]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%cat_1, 1, 2), kwargs = {})
#   %sigmoid : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%sub_1,), kwargs = {})
#   %sub_4 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_6 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub_4), kwargs = {})
#   %sigmoid_1 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_4,), kwargs = {})
#   %sub_5 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_1), kwargs = {})
#   %mul_7 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %sub_5), kwargs = {})
#   %minimum : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.minimum.default](args = (%mul_6, %mul_7), kwargs = {})
#   %full_24 : Tensor "f64[32, s36 + 2048, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, %add_63, 1024], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_13 : Tensor "f64[32, s36, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%full_24, 1, 0, %arg4_1), kwargs = {})
#   %unsqueeze_48 : Tensor "f64[32, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%minimum, 1), kwargs = {})
#   %mul_116 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_48, %arg1_1), kwargs = {})
#   %copy_2 : Tensor "f64[32, s36, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_13, %mul_116), kwargs = {})
#   %slice_scatter_default_2 : Tensor "f64[32, s36 + 2048, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_24, %copy_2, 1, 0, %arg4_1), kwargs = {})
#   %view_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_48, [32, 1024]), kwargs = {})
#   %view_10 : Tensor "f64[32, 1024][3072, 3]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%select_13, [32, 1024]), kwargs = {})
#   %full_25 : Tensor "f64[32, s36 + 2048, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, %add_63, 1024], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_20 : Tensor "f64[32, s36, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%full_25, 1, 0, %arg4_1), kwargs = {})
#   %unsqueeze_49 : Tensor "f64[32, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_9, 1), kwargs = {})
#   %mul_151 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_49, %arg1_1), kwargs = {})
#   %unsqueeze_50 : Tensor "f64[32, 1, 1024][3072, 3072, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_10, 1), kwargs = {})
#   %mul_155 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_50, %arg5_1), kwargs = {})
#   %add_114 : Tensor "f64[32, s36, 1024][1024*s36, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_151, %mul_155), kwargs = {})
#   %copy_4 : Tensor "f64[32, s36, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_20, %add_114), kwargs = {})
#   %slice_scatter_default_3 : Tensor "f64[32, s36 + 2048, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_25, %copy_4, 1, 0, %arg4_1), kwargs = {})
#   return %slice_scatter_default_2,%slice_scatter_default_3
triton_poi_fused_add_copy_minimum_mul_new_zeros_rsub_select_sigmoid_slice_unsqueeze_view_10 = async_compile.triton('triton_poi_fused_add_copy_minimum_mul_new_zeros_rsub_select_sigmoid_slice_unsqueeze_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'in_ptr4': '*fp64', 'in_ptr5': '*fp64', 'out_ptr0': '*fp64', 'out_ptr1': '*fp64', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_minimum_mul_new_zeros_rsub_select_sigmoid_slice_unsqueeze_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2151415808}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy_minimum_mul_new_zeros_rsub_select_sigmoid_slice_unsqueeze_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 1024) % ks0)
    x0 = (xindex % 1024)
    x2 = xindex // ks2
    x3 = (xindex % ks2)
    x4 = xindex
    tmp0 = x1
    tmp1 = ks1
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 1024*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = -tmp3
    tmp6 = libdevice.exp(tmp5)
    tmp7 = tmp4 + tmp6
    tmp8 = (tmp4 / tmp7)
    tmp9 = tl.full([1], 1.0, tl.float64)
    tmp10 = tmp9 - tmp8
    tmp11 = tmp8 * tmp10
    tmp12 = tl.load(in_ptr1 + (x0 + 1024*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp13 = -tmp12
    tmp14 = libdevice.exp(tmp13)
    tmp15 = tmp4 + tmp14
    tmp16 = (tmp4 / tmp15)
    tmp17 = tmp9 - tmp16
    tmp18 = tmp16 * tmp17
    tmp19 = triton_helpers.minimum(tmp11, tmp18)
    tmp20 = tl.load(in_ptr2 + (x3 + 1024*ks1*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 * tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = tl.full([1], 0.0, tl.float64)
    tmp25 = tl.where(tmp2, tmp23, tmp24)
    tmp26 = tl.load(in_ptr3 + (x0 + 1024*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp26 * tmp20
    tmp28 = tl.load(in_ptr4 + (2 + 3*x0 + 3072*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr5 + (x3 + 1024*ks1*x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp27 + tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp2, tmp31, tmp32)
    tmp34 = tl.where(tmp2, tmp33, tmp24)
    tl.store(out_ptr0 + (x4), tmp25, None)
    tl.store(out_ptr1 + (x4), tmp34, None)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/dx/cdxhzomt3vohcktvtqudaf476t2yraa7toyxe3mg4yzfyk32bgcy.py
# Topologically Sorted Source Nodes: [s_lb, sub_2, mul, s_ub, sub_3, mul_1, lam, copy_, sub_5, sub_6, mul_4, sub_7, mu2], Original ATen: [aten.sigmoid, aten.rsub, aten.mul, aten.minimum, aten.slice, aten.diagonal, aten.sub, aten.copy]
# Source node to ATen node mapping:
#   copy_ => copy_3, diagonal_1, slice_17
#   lam => minimum
#   mu2 => mul_11
#   mul => mul_6
#   mul_1 => mul_7
#   mul_4 => mul_10
#   s_lb => sigmoid
#   s_ub => sigmoid_1
#   sub_2 => sub_4
#   sub_3 => sub_5
#   sub_5 => sub_7
#   sub_6 => sub_8
#   sub_7 => sub_9
# Graph fragment:
#   %copy_3 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=copy_3]
#   %slice_scatter_default_2 : Tensor "f64[32, s36 + 2048, 1024][1024*s36 + 2097152, 1024, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_2]
#   %sigmoid : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%sub_1,), kwargs = {})
#   %sub_4 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_6 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub_4), kwargs = {})
#   %sigmoid_1 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_4,), kwargs = {})
#   %sub_5 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid_1), kwargs = {})
#   %mul_7 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %sub_5), kwargs = {})
#   %minimum : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.minimum.default](args = (%mul_6, %mul_7), kwargs = {})
#   %slice_17 : Tensor "f64[32, 1024, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_2, 1, %arg4_1, %add_88), kwargs = {})
#   %diagonal_1 : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.diagonal.default](args = (%slice_17, 0, 1, 2), kwargs = {})
#   %sub_7 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sigmoid_1, %sigmoid), kwargs = {})
#   %sub_8 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %sub_1), kwargs = {})
#   %mul_10 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%minimum, %sub_8), kwargs = {})
#   %sub_9 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %mul_10), kwargs = {})
#   %mul_11 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, 0.5), kwargs = {})
#   %copy_3 : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%diagonal_1, %mul_11), kwargs = {})
#   %slice_tensor : Tensor "f64[32, 1024, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_2, 1, %arg4_1, %add_88), kwargs = {})
#   %diagonal_default : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.diagonal.default](args = (%slice_tensor, 0, 1, 2), kwargs = {})
#   %copy__default : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%diagonal_default, %copy_3), kwargs = {})
#   return %buf78
triton_poi_fused_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_11 = async_compile.triton('triton_poi_fused_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_11', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_11(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (1024*ks0 + 1025*x0 + 2097152*x1 + 1024*ks0*x1), tmp0, None)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_compiled_sigmoid/inductor/6d/c6d4b5ugjl4ledivurdom5ipsnvwtabe4u3ecy5j3lz5mieri25o.py
# Topologically Sorted Source Nodes: [max_1, min_2, evaluation_2, max_3, evaluation_4, max_4, copy__1, neg, corners, maximum_6, maximum_7, errors], Original ATen: [aten.max, aten.min, aten.nan_to_num, aten.slice, aten.diagonal, aten.neg, aten.maximum, aten.view, aten.copy]
# Source node to ATen node mapping:
#   copy__1 => copy_5, diagonal_4, slice_24
#   corners => maximum_6
#   errors => view_7
#   evaluation_2 => eq_6, eq_7, full_default_32, full_default_33, full_default_34, isnan, where_29, where_30, where_31
#   evaluation_4 => eq_8, eq_9, full_default_39, full_default_40, full_default_41, isnan_1, where_33, where_34, where_35
#   max_1 => max_1
#   max_3 => max_2
#   max_4 => max_3
#   maximum_6 => maximum_7
#   maximum_7 => maximum_8
#   min_2 => min_1
#   neg => neg
# Graph fragment:
#   %sub_62 : Tensor "f64[32768, 4][4, 1]cuda:0" = PlaceHolder[target=sub_62]
#   %abs_12 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=abs_12]
#   %maximum_7 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=maximum_7]
#   %abs_14 : Tensor "f64[32768, 3][3, 1]cuda:0" = PlaceHolder[target=abs_14]
#   %copy_5 : Tensor "f64[32, 1024][1024, 1]cuda:0" = PlaceHolder[target=copy_5]
#   %slice_scatter_default_3 : Tensor "f64[32, s36 + 2048, 1024][1024*s36 + 2097152, 1024, 1]cuda:0" = PlaceHolder[target=slice_scatter_default_3]
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%sub_62, 1), kwargs = {})
#   %min_1 : [num_users=1] = call_function[target=torch.ops.aten.min.dim](args = (%sub_62, 1), kwargs = {})
#   %eq_7 : Tensor "b8[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_12, inf), kwargs = {})
#   %full_default_34 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.7976931348623157e+308), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq_6 : Tensor "b8[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_12, -inf), kwargs = {})
#   %full_default_33 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.7976931348623157e+308), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %isnan : Tensor "b8[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%abs_12,), kwargs = {})
#   %full_default_32 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_29 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan, %full_default_32, %abs_12), kwargs = {})
#   %where_30 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_6, %full_default_33, %where_29), kwargs = {})
#   %where_31 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_7, %full_default_34, %where_30), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%where_31, 1), kwargs = {})
#   %eq_9 : Tensor "b8[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_14, inf), kwargs = {})
#   %full_default_41 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.7976931348623157e+308), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq_8 : Tensor "b8[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_14, -inf), kwargs = {})
#   %full_default_40 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1.7976931348623157e+308), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %isnan_1 : Tensor "b8[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%abs_14,), kwargs = {})
#   %full_default_39 : Tensor "f64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_33 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%isnan_1, %full_default_39, %abs_14), kwargs = {})
#   %where_34 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_8, %full_default_40, %where_33), kwargs = {})
#   %where_35 : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_9, %full_default_41, %where_34), kwargs = {})
#   %max_3 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%where_35, 1), kwargs = {})
#   %slice_24 : Tensor "f64[32, 1024, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_3, 1, %add_88, 9223372036854775807), kwargs = {})
#   %diagonal_4 : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.diagonal.default](args = (%slice_24, 0, 1, 2), kwargs = {})
#   %neg : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%getitem_10,), kwargs = {})
#   %maximum_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%getitem_8, %neg), kwargs = {})
#   %maximum_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum_6, %getitem_12), kwargs = {})
#   %maximum_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum_7, %getitem_14), kwargs = {})
#   %view_7 : Tensor "f64[32, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%maximum_8, [32, 1024]), kwargs = {})
#   %copy_5 : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%diagonal_4, %view_7), kwargs = {})
#   %slice_tensor_1 : Tensor "f64[32, 1024, 1024][1024*s36 + 2097152, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_scatter_default_3, 1, %add_88, 9223372036854775807), kwargs = {})
#   %diagonal_default_1 : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.diagonal.default](args = (%slice_tensor_1, 0, 1, 2), kwargs = {})
#   %copy__default_1 : Tensor "f64[32, 1024][1024*s36 + 2097152, 1025]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%diagonal_default_1, %copy_5), kwargs = {})
#   return %maximum_7,%maximum_8,%copy_5,%buf83
triton_poi_fused_copy_diagonal_max_maximum_min_nan_to_num_neg_slice_view_12 = async_compile.triton('triton_poi_fused_copy_diagonal_max_maximum_min_nan_to_num_neg_slice_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'out_ptr1': '*fp64', 'out_ptr3': '*fp64', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_diagonal_max_maximum_min_nan_to_num_neg_slice_view_12', 'mutated_arg_names': ['out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_diagonal_max_maximum_min_nan_to_num_neg_slice_view_12(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 1024)
    x2 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (4*x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3*x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (1 + 3*x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (2 + 3*x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr2 + (3*x0), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr2 + (1 + 3*x0), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr2 + (2 + 3*x0), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = triton_helpers.minimum(tmp0, tmp1)
    tmp8 = triton_helpers.minimum(tmp7, tmp3)
    tmp9 = triton_helpers.minimum(tmp8, tmp5)
    tmp10 = -tmp9
    tmp11 = triton_helpers.maximum(tmp6, tmp10)
    tmp13 = tl.full([1], float("inf"), tl.float64)
    tmp14 = tmp12 == tmp13
    tmp15 = tl.full([1], float("-inf"), tl.float64)
    tmp16 = tmp12 == tmp15
    tmp17 = libdevice.isnan(tmp12).to(tl.int1)
    tmp18 = tl.where(tmp17, tmp15, tmp12)
    tmp19 = tl.full([1], -1.7976931348623157e+308, tl.float64)
    tmp20 = tl.where(tmp16, tmp19, tmp18)
    tmp21 = tl.full([1], 1.7976931348623157e+308, tl.float64)
    tmp22 = tl.where(tmp14, tmp21, tmp20)
    tmp24 = tmp23 == tmp13
    tmp25 = tmp23 == tmp15
    tmp26 = libdevice.isnan(tmp23).to(tl.int1)
    tmp27 = tl.where(tmp26, tmp15, tmp23)
    tmp28 = tl.where(tmp25, tmp19, tmp27)
    tmp29 = tl.where(tmp24, tmp21, tmp28)
    tmp30 = triton_helpers.maximum(tmp22, tmp29)
    tmp32 = tmp31 == tmp13
    tmp33 = tmp31 == tmp15
    tmp34 = libdevice.isnan(tmp31).to(tl.int1)
    tmp35 = tl.where(tmp34, tmp15, tmp31)
    tmp36 = tl.where(tmp33, tmp19, tmp35)
    tmp37 = tl.where(tmp32, tmp21, tmp36)
    tmp38 = triton_helpers.maximum(tmp30, tmp37)
    tmp39 = triton_helpers.maximum(tmp11, tmp38)
    tmp41 = tmp40 == tmp13
    tmp42 = tmp40 == tmp15
    tmp43 = libdevice.isnan(tmp40).to(tl.int1)
    tmp44 = tl.where(tmp43, tmp15, tmp40)
    tmp45 = tl.where(tmp42, tmp19, tmp44)
    tmp46 = tl.where(tmp41, tmp21, tmp45)
    tmp48 = tmp47 == tmp13
    tmp49 = tmp47 == tmp15
    tmp50 = libdevice.isnan(tmp47).to(tl.int1)
    tmp51 = tl.where(tmp50, tmp15, tmp47)
    tmp52 = tl.where(tmp49, tmp19, tmp51)
    tmp53 = tl.where(tmp48, tmp21, tmp52)
    tmp54 = triton_helpers.maximum(tmp46, tmp53)
    tmp56 = tmp55 == tmp13
    tmp57 = tmp55 == tmp15
    tmp58 = libdevice.isnan(tmp55).to(tl.int1)
    tmp59 = tl.where(tmp58, tmp15, tmp55)
    tmp60 = tl.where(tmp57, tmp19, tmp59)
    tmp61 = tl.where(tmp56, tmp21, tmp60)
    tmp62 = triton_helpers.maximum(tmp54, tmp61)
    tmp63 = triton_helpers.maximum(tmp39, tmp62)
    tl.store(out_ptr1 + (x0), tmp63, None)
    tl.store(out_ptr3 + (1048576 + 1024*ks0 + 1025*x1 + 2097152*x2 + 1024*ks0*x2), tmp63, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        s78 = arg0_1
        s36 = arg4_1
        assert_size_stride(arg1_1, (32, s36, 1024), (1024*s36, 1024, 1))
        assert_size_stride(arg2_1, (32, 1024), (1024, 1))
        assert_size_stride(arg3_1, (32, 1024), (1024, 1))
        assert_size_stride(arg5_1, (32, s36, 1024), (1024*s36, 1024, 1))
        assert_size_stride(arg6_1, (32, 1024), (0, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf4 = empty_strided_cuda((32, 1024), (1024, 1), torch.float64)
            buf5 = empty_strided_cuda((32, 1024), (1024, 1), torch.float64)
            # Topologically Sorted Source Nodes: [abs_2, radius_1, ly, uy], Original ATen: [aten.abs, aten.sum, aten.sub, aten.add]
            stream0 = get_raw_stream(0)
            triton_red_fused_abs_add_sub_sum_0.run(arg5_1, arg3_1, buf4, buf5, s36, 32768, s36, stream=stream0)
            buf1 = empty_strided_cuda((32, 1024), (1024, 1), torch.float64)
            buf2 = empty_strided_cuda((32, 1024), (1024, 1), torch.float64)
            buf75 = empty_strided_cuda((32, 1024), (1024, 1), torch.float64)
            buf77 = empty_strided_cuda((32, 1024), (1024, 1), torch.float64)
            # Topologically Sorted Source Nodes: [abs_1, radius, lx, ux, s_lb, sub_2, mul, s_ub, sub_3, mul_1, lam, mul_6, add_2, add_3, mul_2, sub_4, mu1, sig_c, copy_, sub_5, sub_6, mul_4, sub_7, mu2], Original ATen: [aten.abs, aten.sum, aten.sub, aten.add, aten.sigmoid, aten.rsub, aten.mul, aten.minimum, aten.slice, aten.diagonal, aten.copy]
            stream0 = get_raw_stream(0)
            triton_red_fused_abs_add_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_sum_1.run(arg1_1, arg2_1, buf1, buf2, buf75, buf77, s36, 32768, s36, stream=stream0)
            buf6 = empty_strided_cuda((32768, 25, 2), (50, 2, 1), torch.float64)
            # Topologically Sorted Source Nodes: [lx, lx_f, unsqueeze, t, ux, ux_f, sub_8, step, unsqueeze_1, mul_7, out, setitem, unsqueeze_2, getitem_3, sub_9, unsqueeze_3, mul_8, sub_10, repeat_interleave, grid, ly, ly_f, unsqueeze_4, t_1, uy, uy_f, sub_11, step_1, unsqueeze_5, mul_9, out_1, setitem_1, unsqueeze_6, getitem_4, sub_12, unsqueeze_7, mul_10, sub_13, repeat], Original ATen: [aten.sub, aten.view, aten.unsqueeze, aten.arange, aten.add, aten.div, aten.mul, aten.slice, aten.rsub, aten.copy, aten.expand, aten.clone, aten.stack, aten.repeat]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_arange_clone_copy_div_expand_mul_repeat_rsub_slice_stack_sub_unsqueeze_view_2.run(buf2, buf1, buf5, buf4, buf6, 32768, 50, stream=stream0)
            buf9 = empty_strided_cuda((32768, 5), (5, 1), torch.float64)
            buf11 = empty_strided_cuda((32768, 5), (5, 1), torch.float64)
            # Topologically Sorted Source Nodes: [gx, sigmoid_2, sub_14, sigmoid_3, gy, mul_11, targets, z3, getitem_25, sub_53, px, getitem_26, sub_55, py], Original ATen: [aten.select, aten.sigmoid, aten.rsub, aten.mul, aten.view, aten.slice, aten.sub, aten.mean]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mean_mul_rsub_select_sigmoid_slice_sub_view_3.run(buf6, buf9, buf11, 163840, stream=stream0)
            buf14 = empty_strided_cuda((32768, ), (1, ), torch.float64)
            buf7 = empty_strided_cuda((32768, 1), (1, 32768), torch.float64)
            buf10 = empty_strided_cuda((32768, 1), (1, 32768), torch.float64)
            buf17 = reinterpret_tensor(buf10, (32768, ), (1, ), 0); del buf10  # reuse
            buf12 = empty_strided_cuda((32768, 1), (1, 32768), torch.float64)
            buf8 = empty_strided_cuda((32768, 1), (1, 32768), torch.float64)
            buf41 = reinterpret_tensor(buf12, (32768, ), (1, ), 0); del buf12  # reuse
            buf38 = empty_strided_cuda((32768, ), (1, ), torch.bool)
            buf42 = empty_strided_cuda((32768, ), (1, ), torch.bool)
            buf43 = buf42; del buf42  # reuse
            buf39 = buf17; del buf17  # reuse
            buf44 = buf41; del buf41  # reuse
            buf45 = empty_strided_cuda((32768, ), (1, ), torch.float64)
            buf47 = empty_strided_cuda((32768, ), (1, ), torch.float64)
            buf46 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [xs, getitem_9, r, rm, dx, abs_3, m, frexp, ys, getitem_12, r_1, rm_1, dy, abs_4, m_1, frexp_1, gx, sigmoid_2, sub_14, sigmoid_3, gy, mul_11, targets, mean_4, px_1, abs_7, m_3, frexp_2, mean_6, py_1, abs_8, m_4, frexp_3, zbar, zero_1, ones_like, ax, getitem_19, dxn, ones_like_12, sx, getitem_31, truediv_24, mul_69, sum_7, getitem_21, getitem_22, eq, mul_15, sum_5, eq_1, xdeg, one, getitem_10, getitem_11, xbar, abs_5, getitem_13, getitem_14, ybar, abs_6, maximum, maximum_1, mul_13, nxx, truediv_4, sqrt, mul_17, ones_like_1, ay, getitem_20, dyn, mul_14, nyy, truediv_5, sqrt_1, mul_18, maximum_2, m_2, truediv_6, c2, truediv_7, a2, add_9, truediv_8, b2, add_10, truediv_9, p2, add_11, truediv_10, q2, e1, mul_26, mul_27, sub_21, mul_28, add_13, mul_19, add_14, mul_20, add_15, mul_21, e2, df, gt, sub_19, mul_24, add_17, mul_25, mul_22, e3, f, le, ones_like_3, where, truediv_11, zero, where_1, l1, mul_31, mul_32, sub_25, mul_33, df_1, gt_1, sub_23, mul_29, add_19, mul_30, f_1, le_1, ones_like_4, where_2, truediv_12, where_3, l1_1, mul_36, mul_37, sub_29, mul_38, df_2, gt_2, sub_27, mul_34, add_21, mul_35, f_2, le_2, ones_like_5, where_4, truediv_13, where_5, l1_2, mul_41, mul_42, sub_33, mul_43, df_3, gt_3, sub_31, mul_39, add_23, mul_40, f_3, le_3, ones_like_6, where_6, truediv_14, where_7, l1_3, mul_46, mul_47, sub_37, mul_48, df_4, gt_4, sub_35, mul_44, add_25, mul_45, f_4, le_4, ones_like_7, where_8, truediv_15, where_9, l1_4, mul_51, mul_52, sub_41, mul_53, df_5, gt_5, sub_39, mul_49, add_27, mul_50, f_5, le_5, ones_like_8, where_10, truediv_16, where_11, l1_5, mul_56, mul_57, sub_45, mul_58, df_6, gt_6, sub_43, mul_54, add_29, mul_55, f_6, le_6, ones_like_9, where_12, truediv_17, where_13, l1_6, mul_61, mul_62, sub_49, mul_63, df_7, gt_7, sub_47, mul_59, add_31, mul_60, f_7, le_7, ones_like_10, where_14, truediv_18, where_15, l1_7, prod, sub_51, truediv_20, total, mul_64, mul_65, sub_52, disc, sqrt_2, add_33, l2, mul_67, drop2, gt_8, le_8, ones_like_11, where_16, truediv_21, l3, mul_68, drop3, add_34, tx, add_35, ty, pick_x, and_, cx, or__4, ones_like_14, where_18, truediv_25, mul_70, ones_like_15, where_19, truediv_26, a0, mul_73, sub_57, ones_like_13, sy, getitem_32, truediv_27, mul_71, sum_8, getitem_23, getitem_24, eq_2, mul_16, sum_6, eq_3, ydeg, invert, and__1, cy, or__5, ones_like_16, where_21, truediv_28, mul_72, ones_like_17, where_22, truediv_29, b0, mul_74, k, ones_like_18, zero_2, u, abs_9, v, abs_10, maximum_4, m_5, km, inv, mul_75, un, mul_76, add_36, vn, mul_77, denom], Original ATen: [aten.slice, aten.select, aten.sub, aten.mean, aten.abs, aten.amax, aten.frexp, aten.sigmoid, aten.rsub, aten.mul, aten.zeros_like, aten.ones_like, aten.ldexp, aten.unsqueeze, aten.div, aten.sum, aten.eq, aten.bitwise_or, aten.add, aten.maximum, aten.sqrt, aten.pow, aten.gt, aten.le, aten.where, aten.clamp_min, aten.bitwise_and, aten.bitwise_not, aten.reciprocal]
            stream0 = get_raw_stream(0)
            triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_rsub_select_sigmoid_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_4.run(buf39, buf44, buf43, buf46, buf6, buf9, buf11, buf7, buf8, buf38, buf45, buf47, 32768, 25, stream=stream0)
            del buf11
            del buf9
            buf48 = empty_strided_cuda((32768, 3), (3, 1), torch.float64)
            # Topologically Sorted Source Nodes: [xs, ys, getitem_10, getitem_11, xbar, getitem_13, getitem_14, ybar, zero_2, u, v, inv, mul_78, un, vn, C, coef, mul_79, truediv_35, A, mul_80, truediv_36, B], Original ATen: [aten.slice, aten.select, aten.add, aten.zeros_like, aten.where, aten.reciprocal, aten.mul, aten.div, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_5.run(buf46, buf45, buf47, buf38, buf6, buf7, buf39, buf43, buf8, buf44, buf48, 32768, 3, stream=stream0)
            del buf38
            del buf43
            del buf6
            buf49 = reinterpret_tensor(buf8, (32768, ), (1, ), 0); del buf8  # reuse
            buf56 = reinterpret_tensor(buf7, (32768, ), (1, ), 0); del buf7  # reuse
            buf58 = buf47; del buf47  # reuse
            buf60 = buf46; del buf46  # reuse
            buf67 = buf45; del buf45  # reuse
            buf69 = buf44; del buf44  # reuse
            buf71 = buf39; del buf39  # reuse
            buf79 = empty_strided_cuda((32, 1024), (1024, 1), torch.float64)
            buf85 = empty_strided_cuda((4, ), (1, ), torch.bool)
            # Topologically Sorted Source Nodes: [lx, lx_f, ux, ux_f, ly, ly_f, uy, uy_f, getitem_33, reshape_5, As, abc, Bs, Cs, A_2, A_by_ly, isinf, setitem_2, u_1, le_12, l, ge, logical_and, guard, setitem_4, arg, abs_11, le_14, clamp, acos, truediv_39, full_like, where_28, sub_61, cos, mul_86, x1, sub_64, truediv_40, root1, le_15, ge_2, logical_and_2, guard_2, setitem_6, sub_62, cos_1, mul_87, x2, sub_65, truediv_41, root2, le_16, ge_3, logical_and_3, guard_3, setitem_7, sub_63, cos_2, mul_88, x3, sub_66, truediv_42, root3, le_17, ge_4, logical_and_4, guard_4, setitem_8, A_by_uy, isinf_1, setitem_3, le_13, ge_1, logical_and_1, guard_1, setitem_5, arg_1, abs_13, le_18, clamp_1, acos_1, truediv_43, full_like_1, where_29, sub_69, cos_3, mul_94, x1_1, sub_72, truediv_44, root1_uy, le_19, ge_5, logical_and_5, guard_5, setitem_9, sub_70, cos_4, mul_95, x2_1, sub_73, truediv_45, root2_uy, le_20, ge_6, logical_and_6, guard_6, setitem_10, sub_71, cos_5, mul_96, x3_1, sub_74, truediv_46, root3_uy, le_21, ge_7, logical_and_7, guard_7, setitem_11, As_1, mul_101, Bs_1, mul_102, add_51, Cs_1, dual_c, gt_9, lt, or__6, a_nonzero], Original ATen: [aten.sub, aten.view, aten.add, aten.select, aten.clone, aten._unsafe_view, aten.stack, aten.div, aten.isinf, aten.lift_fresh, aten.index_put, aten.neg, aten.le, aten.ge, aten.logical_and, aten.logical_not, aten.mul, aten.abs, aten.clamp, aten.acos, aten.full_like, aten.where, aten.cos, aten.rsub, aten.log, aten.gt, aten.lt, aten.bitwise_or, aten.all]
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_abs_acos_add_all_bitwise_or_clamp_clone_cos_div_full_like_ge_gt_index_put_isinf_le_lift_fresh_log_logical_and_logical_not_lt_mul_neg_rsub_select_stack_sub_view_where_6.run(buf48, arg6_1, buf4, buf5, buf2, buf1, arg2_1, arg3_1, buf49, buf56, buf58, buf60, buf67, buf69, buf71, buf79, buf85, 4, 8192, stream=stream0)
            del arg2_1
            del arg3_1
            del arg6_1
            buf86 = empty_strided_cuda((), (), torch.bool)
            buf88 = buf86; del buf86  # reuse
            # Topologically Sorted Source Nodes: [gt_9, lt, or__6, a_nonzero], Original ATen: [aten.gt, aten.lt, aten.bitwise_or, aten.all]
            stream0 = get_raw_stream(0)
            triton_per_fused_all_bitwise_or_gt_lt_7.run(buf88, buf85, 1, 4, stream=stream0)
            del buf85
            buf61 = empty_strided_cuda((32768, 3), (3, 1), torch.float64)
            buf62 = buf61; del buf61  # reuse
            buf63 = buf62; del buf62  # reuse
            buf72 = empty_strided_cuda((32768, 3), (3, 1), torch.float64)
            buf73 = buf72; del buf72  # reuse
            buf74 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [ly, ly_f, uy, uy_f, abc, Bs, Cs, A_2, stack_5, sig_1, sub_67, roots_y, mul_89, sigmoid_prime_y_1, unsqueeze_12, mul_91, B_2, unsqueeze_13, mul_92, add_44, C_2, unsqueeze_14, planar_1, sub_68, evaluation_1, stack_6, sig_2, sub_75, roots_y_uy, mul_97, sigmoid_prime_y_2, unsqueeze_16, mul_99, unsqueeze_17, mul_100, add_49, unsqueeze_18, planar_2, sub_76, evaluation_3], Original ATen: [aten.sub, aten.view, aten.add, aten.stack, aten.select, aten.sigmoid, aten.rsub, aten.unsqueeze, aten.mul, aten.abs]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_8.run(buf63, buf74, buf49, buf48, buf56, buf58, buf60, buf4, buf67, buf69, buf71, buf5, 32768, 3, stream=stream0)
            del buf56
            del buf58
            del buf60
            del buf67
            del buf69
            del buf71
            buf13 = empty_strided_cuda((32768, 4), (4, 1), torch.float64)
            buf52 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [lx, lx_f, ux, ux_f, ly, ly_f, uy, uy_f, xs_1, sig, sub_59, ys_1, mul_81, sigmoid_prime_y, abc, Bs, Cs, A_1, unsqueeze_8, mul_83, B_1, unsqueeze_9, mul_84, add_39, C_1, unsqueeze_10, planar, evaluation], Original ATen: [aten.sub, aten.view, aten.add, aten.stack, aten.sigmoid, aten.rsub, aten.mul, aten.select, aten.unsqueeze]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_9.run(buf52, buf1, buf2, buf4, buf5, buf49, buf48, 32768, 4, stream=stream0)
            ps0 = 2048 + s36
            ps1 = 2097152 + 1024*s36
            buf76 = empty_strided_cuda((32, 2048 + s36, 1024), (2097152 + 1024*s36, 1024, 1), torch.float64)
            buf80 = empty_strided_cuda((32, 2048 + s36, 1024), (2097152 + 1024*s36, 1024, 1), torch.float64)
            # Topologically Sorted Source Nodes: [Bs, s_lb, sub_2, mul, s_ub, sub_3, mul_1, lam, new_rg, setitem_12, unsqueeze_19, mul_103, As_1, Bs_1, new_dg, setitem_13, unsqueeze_20, mul_104, unsqueeze_21, mul_105, add_56], Original ATen: [aten.select, aten.sigmoid, aten.rsub, aten.mul, aten.minimum, aten.new_zeros, aten.slice, aten.unsqueeze, aten.copy, aten.view, aten.add]
            triton_poi_fused_add_copy_minimum_mul_new_zeros_rsub_select_sigmoid_slice_unsqueeze_view_10_xnumel = 67108864 + 32768*s36
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy_minimum_mul_new_zeros_rsub_select_sigmoid_slice_unsqueeze_view_10.run(buf1, buf2, arg1_1, buf49, buf48, arg5_1, buf76, buf80, ps0, s36, ps1, triton_poi_fused_add_copy_minimum_mul_new_zeros_rsub_select_sigmoid_slice_unsqueeze_view_10_xnumel, stream=stream0)
            del arg1_1
            del arg5_1
            # Topologically Sorted Source Nodes: [s_lb, sub_2, mul, s_ub, sub_3, mul_1, lam, copy_, sub_5, sub_6, mul_4, sub_7, mu2], Original ATen: [aten.sigmoid, aten.rsub, aten.mul, aten.minimum, aten.slice, aten.diagonal, aten.sub, aten.copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_11.run(buf77, buf76, s36, 32768, stream=stream0)
            buf87 = reinterpret_tensor(buf77, (32768, ), (1, ), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [max_1, min_2, evaluation_2, max_3, evaluation_4, max_4, copy__1, neg, corners, maximum_6, maximum_7, errors], Original ATen: [aten.max, aten.min, aten.nan_to_num, aten.slice, aten.diagonal, aten.neg, aten.maximum, aten.view, aten.copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy_diagonal_max_maximum_min_nan_to_num_neg_slice_view_12.run(buf52, buf63, buf74, buf87, buf80, s36, 32768, stream=stream0)
            del buf52
            del buf63
            del buf74
        return (buf75, buf76, buf79, buf80, buf88, buf49, reinterpret_tensor(buf48, (32768, ), (3, ), 2), reinterpret_tensor(buf48, (32768, ), (3, ), 0), reinterpret_tensor(buf1, (32768, ), (1, ), 0), reinterpret_tensor(buf2, (32768, ), (1, ), 0), reinterpret_tensor(buf4, (32768, ), (1, ), 0), reinterpret_tensor(buf5, (32768, ), (1, ), 0), buf87, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 2
    arg1_1 = rand_strided((32, 2, 1024), (2048, 1024, 1), device='cuda:0', dtype=torch.float64)
    arg2_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.float64)
    arg3_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.float64)
    arg4_1 = 2
    arg5_1 = rand_strided((32, 2, 1024), (2048, 1024, 1), device='cuda:0', dtype=torch.float64)
    arg6_1 = rand_strided((32, 1024), (0, 1), device='cuda:0', dtype=torch.float64)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
