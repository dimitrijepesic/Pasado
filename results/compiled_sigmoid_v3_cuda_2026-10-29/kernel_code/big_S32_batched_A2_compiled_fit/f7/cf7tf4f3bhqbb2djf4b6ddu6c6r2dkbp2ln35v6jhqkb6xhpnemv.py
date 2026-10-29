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


# kernel path: /content/inductor_caches/logs_big_S32_batched_A2_compiled_fit/inductor/nv/cnvgyx5byqj5anckxt5s4tf4k4gu2as4z27dgwbx5stitt5jhjos.py
# Topologically Sorted Source Nodes: [z3, getitem_18, sub_38, px, getitem_19, sub_40, py], Original ATen: [aten.view, aten.slice, aten.sub, aten.mean]
# Source node to ATen node mapping:
#   getitem_18 => slice_5
#   getitem_19 => slice_6
#   px => mean_3
#   py => mean_5
#   sub_38 => sub_38
#   sub_40 => sub_40
#   z3 => view
# Graph fragment:
#   %arg0_1 : Tensor "f64[32768, 25][25, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %view : Tensor "f64[32768, 5, 5][25, 5, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [32768, 5, 5]), kwargs = {})
#   %slice_5 : Tensor "f64[32768, 1, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view, 1, 0, 1), kwargs = {})
#   %sub_38 : Tensor "f64[32768, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %slice_5), kwargs = {})
#   %mean_3 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_38, [2]), kwargs = {})
#   %slice_6 : Tensor "f64[32768, 5, 1][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view, 2, 0, 1), kwargs = {})
#   %sub_40 : Tensor "f64[32768, 5, 5][25, 5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %slice_6), kwargs = {})
#   %mean_5 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_40, [1]), kwargs = {})
#   return %mean_3,%mean_5
triton_poi_fused_mean_slice_sub_view_0 = async_compile.triton('triton_poi_fused_mean_slice_sub_view_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_slice_sub_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 19, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_slice_sub_view_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 5
    x0 = (xindex % 5)
    tmp0 = tl.load(in_ptr0 + (5*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (25*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 5*x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 25*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 5*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 25*x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 5*x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (3 + 25*x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (4 + 5*x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (4 + 25*x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (x0 + 25*x1), None)
    tmp23 = tl.load(in_ptr0 + (5 + x0 + 25*x1), None)
    tmp24 = tl.load(in_ptr0 + (5 + 25*x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (10 + x0 + 25*x1), None)
    tmp28 = tl.load(in_ptr0 + (10 + 25*x1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (15 + x0 + 25*x1), None)
    tmp32 = tl.load(in_ptr0 + (15 + 25*x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (20 + x0 + 25*x1), None)
    tmp36 = tl.load(in_ptr0 + (20 + 25*x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 - tmp12
    tmp14 = tmp10 + tmp13
    tmp17 = tmp15 - tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tl.full([1], 5.0, tl.float64)
    tmp20 = (tmp18 / tmp19)
    tmp22 = tmp21 - tmp1
    tmp25 = tmp23 - tmp24
    tmp26 = tmp22 + tmp25
    tmp29 = tmp27 - tmp28
    tmp30 = tmp26 + tmp29
    tmp33 = tmp31 - tmp32
    tmp34 = tmp30 + tmp33
    tmp37 = tmp35 - tmp36
    tmp38 = tmp34 + tmp37
    tmp39 = (tmp38 / tmp19)
    tl.store(out_ptr0 + (x2), tmp20, None)
    tl.store(out_ptr1 + (x2), tmp39, None)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_A2_compiled_fit/inductor/jk/cjkvuiojmcfk4hyw6v7jyteudbfmilrxx3uwn4h5c53gr65a26la.py
# Topologically Sorted Source Nodes: [xs, getitem_2, r, rm, dx, abs_1, m, frexp, ys, getitem_5, r_1, rm_1, dy, abs_2, m_1, frexp_1, mean_4, px_1, abs_5, m_3, frexp_2, mean_6, py_1, abs_6, m_4, frexp_3, zbar, zero_1, ones_like, ax, getitem_12, dxn, ones_like_12, sx, getitem_24, truediv_22, mul_56, sum_5, getitem_14, getitem_15, eq, mul_2, sum_3, eq_1, xdeg, one, getitem_3, getitem_4, xbar, abs_3, getitem_6, getitem_7, ybar, abs_4, maximum, maximum_1, mul, nxx, truediv_2, sqrt, mul_4, ones_like_1, ay, getitem_13, dyn, mul_1, nyy, truediv_3, sqrt_1, mul_5, maximum_2, m_2, truediv_4, c2, truediv_5, a2, add_2, truediv_6, b2, add_3, truediv_7, p2, add_4, truediv_8, q2, e1, mul_13, mul_14, sub_6, mul_15, add_6, mul_6, add_7, mul_7, add_8, mul_8, e2, df, gt, sub_4, mul_11, add_10, mul_12, mul_9, e3, f, le, ones_like_3, where, truediv_9, zero, where_1, l1, mul_18, mul_19, sub_10, mul_20, df_1, gt_1, sub_8, mul_16, add_12, mul_17, f_1, le_1, ones_like_4, where_2, truediv_10, where_3, l1_1, mul_23, mul_24, sub_14, mul_25, df_2, gt_2, sub_12, mul_21, add_14, mul_22, f_2, le_2, ones_like_5, where_4, truediv_11, where_5, l1_2, mul_28, mul_29, sub_18, mul_30, df_3, gt_3, sub_16, mul_26, add_16, mul_27, f_3, le_3, ones_like_6, where_6, truediv_12, where_7, l1_3, mul_33, mul_34, sub_22, mul_35, df_4, gt_4, sub_20, mul_31, add_18, mul_32, f_4, le_4, ones_like_7, where_8, truediv_13, where_9, l1_4, mul_38, mul_39, sub_26, mul_40, df_5, gt_5, sub_24, mul_36, add_20, mul_37, f_5, le_5, ones_like_8, where_10, truediv_14, where_11, l1_5, mul_43, mul_44, sub_30, mul_45, df_6, gt_6, sub_28, mul_41, add_22, mul_42, f_6, le_6, ones_like_9, where_12, truediv_15, where_13, l1_6, mul_48, mul_49, sub_34, mul_50, df_7, gt_7, sub_32, mul_46, add_24, mul_47, f_7, le_7, ones_like_10, where_14, truediv_16, where_15, l1_7, prod, sub_36, truediv_18, total, mul_51, mul_52, sub_37, disc, sqrt_2, add_26, l2, mul_54, drop2, gt_8, le_8, ones_like_11, where_16, truediv_19, l3, mul_55, drop3, add_27, tx, add_28, ty, pick_x, and_, cx, or__4, ones_like_14, where_18, truediv_23, mul_57, ones_like_15, where_19, truediv_24, a0, mul_60, sub_42, ones_like_13, sy, getitem_25, truediv_25, mul_58, sum_6, getitem_16, getitem_17, eq_2, mul_3, sum_4, eq_3, ydeg, invert, and__1, cy, or__5, ones_like_16, where_21, truediv_26, mul_59, ones_like_17, where_22, truediv_27, b0, mul_61, k, ones_like_18, zero_2, u, abs_7, v, abs_8, maximum_4, m_5, km, inv, mul_62, un, mul_63, add_29, vn, mul_64, denom], Original ATen: [aten.slice, aten.select, aten.sub, aten.mean, aten.abs, aten.amax, aten.frexp, aten.zeros_like, aten.ones_like, aten.ldexp, aten.unsqueeze, aten.div, aten.mul, aten.sum, aten.eq, aten.bitwise_or, aten.add, aten.maximum, aten.sqrt, aten.pow, aten.gt, aten.le, aten.where, aten.clamp_min, aten.bitwise_and, aten.bitwise_not, aten.reciprocal]
# Source node to ATen node mapping:
#   a0 => where_20
#   a2 => pow_2
#   abs_1 => abs_1
#   abs_2 => abs_2
#   abs_3 => abs_3
#   abs_4 => abs_4
#   abs_5 => abs_5
#   abs_6 => abs_6
#   abs_7 => abs_7
#   abs_8 => abs_8
#   add_10 => add_10
#   add_12 => add_12
#   add_14 => add_14
#   add_16 => add_16
#   add_18 => add_18
#   add_2 => add_2
#   add_20 => add_20
#   add_22 => add_22
#   add_24 => add_24
#   add_26 => add_26
#   add_27 => add_27
#   add_28 => add_28
#   add_29 => add_29
#   add_3 => add_3
#   add_4 => add_4
#   add_6 => add_6
#   add_7 => add_7
#   add_8 => add_8
#   and_ => bitwise_and
#   and__1 => bitwise_and_1
#   ax => ldexp
#   ay => ldexp_1
#   b0 => where_23
#   b2 => pow_3
#   c2 => pow_1
#   cx => bitwise_or_2
#   cy => bitwise_or_3
#   denom => add_30
#   df => add_11
#   df_1 => add_13
#   df_2 => add_15
#   df_3 => add_17
#   df_4 => add_19
#   df_5 => add_21
#   df_6 => add_23
#   df_7 => add_25
#   disc => clamp_min_1
#   drop2 => le_9
#   drop3 => le_10
#   dx => sub_1
#   dxn => div
#   dy => sub_3
#   dyn => div_1
#   e1 => add_5
#   e2 => add_9
#   e3 => mul_10
#   eq => eq
#   eq_1 => eq_1
#   eq_2 => eq_2
#   eq_3 => eq_3
#   f => sub_5
#   f_1 => sub_9
#   f_2 => sub_13
#   f_3 => sub_17
#   f_4 => sub_21
#   f_5 => sub_25
#   f_6 => sub_29
#   f_7 => sub_33
#   frexp => frexp
#   frexp_1 => frexp_1
#   frexp_2 => frexp_2
#   frexp_3 => frexp_3
#   getitem_12 => unsqueeze
#   getitem_13 => unsqueeze_1
#   getitem_14 => select_6
#   getitem_15 => select_7
#   getitem_16 => select_8
#   getitem_17 => select_9
#   getitem_2 => slice_3
#   getitem_24 => unsqueeze_2
#   getitem_25 => unsqueeze_3
#   getitem_3 => select_2
#   getitem_4 => select_3
#   getitem_5 => slice_4
#   getitem_6 => select_4
#   getitem_7 => select_5
#   gt => gt
#   gt_1 => gt_1
#   gt_2 => gt_2
#   gt_3 => gt_3
#   gt_4 => gt_4
#   gt_5 => gt_5
#   gt_6 => gt_6
#   gt_7 => gt_7
#   gt_8 => gt_8
#   inv => mul_62, reciprocal
#   invert => bitwise_not
#   k => sub_43
#   km => div_30
#   l1 => sub_7
#   l1_1 => sub_11
#   l1_2 => sub_15
#   l1_3 => sub_19
#   l1_4 => sub_23
#   l1_5 => sub_27
#   l1_6 => sub_31
#   l1_7 => sub_35
#   l2 => mul_53
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
#   mul => mul
#   mul_1 => mul_1
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_2 => mul_2
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_29 => mul_29
#   mul_3 => mul_3
#   mul_30 => mul_30
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_33 => mul_33
#   mul_34 => mul_34
#   mul_35 => mul_35
#   mul_36 => mul_36
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   mul_4 => mul_4
#   mul_40 => mul_40
#   mul_41 => mul_41
#   mul_42 => mul_42
#   mul_43 => mul_43
#   mul_44 => mul_44
#   mul_45 => mul_45
#   mul_46 => mul_46
#   mul_47 => mul_47
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_5 => mul_5
#   mul_50 => mul_50
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_54 => mul_54
#   mul_55 => mul_55
#   mul_56 => mul_56
#   mul_57 => mul_57
#   mul_58 => mul_58
#   mul_59 => mul_59
#   mul_6 => mul_6
#   mul_60 => mul_60
#   mul_61 => mul_61
#   mul_62 => mul_63
#   mul_63 => mul_64
#   mul_64 => mul_65
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   nxx => sum_1
#   nyy => sum_2
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
#   prod => div_17
#   px_1 => sub_39
#   py_1 => sub_41
#   q2 => pow_5
#   r => sub
#   r_1 => sub_2
#   rm => mean
#   rm_1 => mean_1
#   sqrt => sqrt
#   sqrt_1 => sqrt_1
#   sqrt_2 => sqrt_2
#   sub_10 => sub_10
#   sub_12 => sub_12
#   sub_14 => sub_14
#   sub_16 => sub_16
#   sub_18 => sub_18
#   sub_20 => sub_20
#   sub_22 => sub_22
#   sub_24 => sub_24
#   sub_26 => sub_26
#   sub_28 => sub_28
#   sub_30 => sub_30
#   sub_32 => sub_32
#   sub_34 => sub_34
#   sub_36 => sub_36
#   sub_37 => sub_37
#   sub_4 => sub_4
#   sub_42 => sub_42
#   sub_6 => sub_6
#   sub_8 => sub_8
#   sum_3 => sum_3
#   sum_4 => sum_4
#   sum_5 => sum_5
#   sum_6 => sum_6
#   sx => ldexp_2
#   sy => ldexp_3
#   total => clamp_min
#   truediv_10 => div_10
#   truediv_11 => div_11
#   truediv_12 => div_12
#   truediv_13 => div_13
#   truediv_14 => div_14
#   truediv_15 => div_15
#   truediv_16 => div_16
#   truediv_18 => div_18
#   truediv_19 => div_19
#   truediv_2 => div_2
#   truediv_22 => div_22
#   truediv_23 => div_23
#   truediv_24 => div_24
#   truediv_25 => div_25
#   truediv_26 => div_26
#   truediv_27 => div_27
#   truediv_3 => div_3
#   truediv_4 => div_4
#   truediv_5 => div_5
#   truediv_6 => div_6
#   truediv_7 => div_7
#   truediv_8 => div_8
#   truediv_9 => div_9
#   tx => div_20
#   ty => div_21
#   u => where_24
#   un => div_28
#   v => where_25
#   vn => div_29
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
#   xbar => add
#   xdeg => bitwise_or
#   xs => select, slice_1
#   ybar => add_1
#   ydeg => bitwise_or_1
#   ys => select_1, slice_2
#   zbar => mean_2
#   zero => full_default_3
#   zero_1 => full_default_15
#   zero_2 => full_default_20
# Graph fragment:
#   %arg0_1 : Tensor "f64[32768, 25][25, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f64[32768, 25, 2][50, 2, 1]cuda:0" = PlaceHolder[target=arg1_1]
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
#   %sum_1 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sum_1]
#   %sum_2 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sum_2]
#   %maximum_3 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=maximum_3]
#   %add_8 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_8]
#   %add_5 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_5]
#   %add_9 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_9]
#   %mul_10 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=mul_10]
#   %div_10 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_10]
#   %sub_11 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sub_11]
#   %div_12 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_12]
#   %where_7 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_7]
#   %sub_19 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sub_19]
#   %div_14 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_14]
#   %where_11 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_11]
#   %sub_27 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sub_27]
#   %div_16 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_16]
#   %where_15 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_15]
#   %sub_35 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sub_35]
#   %le_11 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=le_11]
#   %bitwise_or_3 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_3]
#   %bitwise_or_4 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_4]
#   %sum_5 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sum_5]
#   %bitwise_or_5 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_5]
#   %sum_6 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=sum_6]
#   %maximum_5 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=maximum_5]
#   %buf6 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=buf6]
#   %where_20 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_20]
#   %where_23 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_23]
#   %slice_1 : Tensor "f64[32768, 5, 2][50, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 0, 9223372036854775807, 5), kwargs = {})
#   %select : Tensor "f64[32768, 5][50, 10]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%slice_1, 2, 0), kwargs = {})
#   %slice_3 : Tensor "f64[32768, 1][50, 10]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select, 1, 0, 1), kwargs = {})
#   %sub : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select, %slice_3), kwargs = {})
#   %mean : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub, [1], True), kwargs = {})
#   %sub_1 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %mean), kwargs = {})
#   %abs_1 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %amax : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_1, [1]), kwargs = {})
#   %frexp : [num_users=1] = call_function[target=torch.ops.aten.frexp.Tensor](args = (%amax,), kwargs = {})
#   %slice_2 : Tensor "f64[32768, 5, 2][50, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 0, 5), kwargs = {})
#   %select_1 : Tensor "f64[32768, 5][50, 2]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%slice_2, 2, 1), kwargs = {})
#   %slice_4 : Tensor "f64[32768, 1][50, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select_1, 1, 0, 1), kwargs = {})
#   %sub_2 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_1, %slice_4), kwargs = {})
#   %mean_1 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%sub_2, [1], True), kwargs = {})
#   %sub_3 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_2, %mean_1), kwargs = {})
#   %abs_2 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_3,), kwargs = {})
#   %amax_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_2, [1]), kwargs = {})
#   %frexp_1 : [num_users=1] = call_function[target=torch.ops.aten.frexp.Tensor](args = (%amax_1,), kwargs = {})
#   %mean_4 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mean_3, [1], True), kwargs = {})
#   %sub_39 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_3, %mean_4), kwargs = {})
#   %abs_5 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_39,), kwargs = {})
#   %amax_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_5, [1]), kwargs = {})
#   %frexp_2 : [num_users=1] = call_function[target=torch.ops.aten.frexp.Tensor](args = (%amax_2,), kwargs = {})
#   %mean_6 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mean_5, [1], True), kwargs = {})
#   %sub_41 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_5, %mean_6), kwargs = {})
#   %abs_6 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_41,), kwargs = {})
#   %amax_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_6, [1]), kwargs = {})
#   %frexp_3 : [num_users=1] = call_function[target=torch.ops.aten.frexp.Tensor](args = (%amax_3,), kwargs = {})
#   %mean_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%arg0_1, [1]), kwargs = {})
#   %full_default_15 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32768], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ldexp : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ldexp.Tensor](args = (%full_default, %getitem_1), kwargs = {})
#   %unsqueeze : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%ldexp, 1), kwargs = {})
#   %div : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %unsqueeze), kwargs = {})
#   %full_default_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ldexp_2 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ldexp.Tensor](args = (%full_default_13, %getitem_5), kwargs = {})
#   %unsqueeze_2 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%ldexp_2, 1), kwargs = {})
#   %div_22 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_39, %unsqueeze_2), kwargs = {})
#   %mul_56 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %div_22), kwargs = {})
#   %sum_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_56, [1]), kwargs = {})
#   %select_6 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select, 1, 0), kwargs = {})
#   %select_7 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select, 1, -1), kwargs = {})
#   %eq : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_6, %select_7), kwargs = {})
#   %mul_2 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %sub_1), kwargs = {})
#   %sum_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [1]), kwargs = {})
#   %eq_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_3, 0), kwargs = {})
#   %bitwise_or : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%eq, %eq_1), kwargs = {})
#   %full_default_2 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_2 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select, 1, 0), kwargs = {})
#   %select_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%mean, 1, 0), kwargs = {})
#   %add : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, %select_3), kwargs = {})
#   %abs_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add,), kwargs = {})
#   %select_4 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_1, 1, 0), kwargs = {})
#   %select_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%mean_1, 1, 0), kwargs = {})
#   %add_1 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %abs_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_1,), kwargs = {})
#   %maximum : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%abs_3, %abs_4), kwargs = {})
#   %maximum_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%full_default_2, %maximum), kwargs = {})
#   %mul : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %div), kwargs = {})
#   %sum_1 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %div_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, 5), kwargs = {})
#   %sqrt : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%div_2,), kwargs = {})
#   %mul_4 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%ldexp, %sqrt), kwargs = {})
#   %full_default_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ldexp_1 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ldexp.Tensor](args = (%full_default_1, %getitem_3), kwargs = {})
#   %unsqueeze_1 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%ldexp_1, 1), kwargs = {})
#   %div_1 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, %unsqueeze_1), kwargs = {})
#   %mul_1 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %div_1), kwargs = {})
#   %sum_2 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
#   %div_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_2, 5), kwargs = {})
#   %sqrt_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%div_3,), kwargs = {})
#   %mul_5 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%ldexp_1, %sqrt_1), kwargs = {})
#   %maximum_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%mul_4, %mul_5), kwargs = {})
#   %maximum_3 : Tensor "f64[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum_1, %maximum_2), kwargs = {})
#   %div_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%full_default_2, %maximum_3), kwargs = {})
#   %pow_1 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_4, 2), kwargs = {})
#   %div_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add, %maximum_3), kwargs = {})
#   %pow_2 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_5, 2), kwargs = {})
#   %add_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %div_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1, %maximum_3), kwargs = {})
#   %pow_3 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_6, 2), kwargs = {})
#   %add_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %pow_3), kwargs = {})
#   %div_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_4, %maximum_3), kwargs = {})
#   %pow_4 : Tensor "f64[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_7, 2), kwargs = {})
#   %add_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %pow_4), kwargs = {})
#   %div_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_5, %maximum_3), kwargs = {})
#   %pow_5 : Tensor "f64[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_8, 2), kwargs = {})
#   %add_5 : Tensor "f64[32768][1]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %pow_5), kwargs = {})
#   %mul_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 3.0), kwargs = {})
#   %mul_14 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 2.0), kwargs = {})
#   %sub_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %mul_15 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %add_5), kwargs = {})
#   %add_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_3), kwargs = {})
#   %mul_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_4, %add_6), kwargs = {})
#   %add_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %mul_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_5, %add_7), kwargs = {})
#   %add_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %mul_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_4, %pow_5), kwargs = {})
#   %add_9 : Tensor "f64[32768][1]cuda:0"[num_users=17] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %mul_8), kwargs = {})
#   %add_11 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %add_9), kwargs = {})
#   %gt : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_11, 0), kwargs = {})
#   %sub_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_5), kwargs = {})
#   %mul_11 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %add_5), kwargs = {})
#   %add_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %add_9), kwargs = {})
#   %mul_12 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %add_5), kwargs = {})
#   %mul_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, %pow_4), kwargs = {})
#   %mul_10 : Tensor "f64[32768][1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %pow_5), kwargs = {})
#   %sub_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_12, %mul_10), kwargs = {})
#   %le : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_11, 0), kwargs = {})
#   %full_default_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default_4, %add_11), kwargs = {})
#   %div_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, %where), kwargs = {})
#   %full_default_3 : Tensor "f64[32768][1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([32768], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %div_9, %full_default_3), kwargs = {})
#   %sub_7 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %where_1), kwargs = {})
#   %mul_18 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 3.0), kwargs = {})
#   %mul_19 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 2.0), kwargs = {})
#   %sub_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_18, %mul_19), kwargs = {})
#   %mul_20 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %sub_7), kwargs = {})
#   %add_13 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %add_9), kwargs = {})
#   %gt_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_13, 0), kwargs = {})
#   %sub_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %add_5), kwargs = {})
#   %mul_16 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %sub_7), kwargs = {})
#   %add_12 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %add_9), kwargs = {})
#   %mul_17 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sub_7), kwargs = {})
#   %sub_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_17, %mul_10), kwargs = {})
#   %le_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_13, 0), kwargs = {})
#   %full_default_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default_5, %add_13), kwargs = {})
#   %div_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_9, %where_2), kwargs = {})
#   %where_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %div_10, %full_default_3), kwargs = {})
#   %sub_11 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %where_3), kwargs = {})
#   %mul_23 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, 3.0), kwargs = {})
#   %mul_24 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 2.0), kwargs = {})
#   %sub_14 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_23, %mul_24), kwargs = {})
#   %mul_25 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %sub_11), kwargs = {})
#   %add_15 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %add_9), kwargs = {})
#   %gt_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_15, 0), kwargs = {})
#   %sub_12 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_11, %add_5), kwargs = {})
#   %mul_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %sub_11), kwargs = {})
#   %add_14 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %add_9), kwargs = {})
#   %mul_22 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %sub_11), kwargs = {})
#   %sub_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_22, %mul_10), kwargs = {})
#   %le_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_15, 0), kwargs = {})
#   %full_default_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %full_default_6, %add_15), kwargs = {})
#   %div_11 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_13, %where_4), kwargs = {})
#   %where_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %div_11, %full_default_3), kwargs = {})
#   %sub_15 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_11, %where_5), kwargs = {})
#   %mul_28 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, 3.0), kwargs = {})
#   %mul_29 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 2.0), kwargs = {})
#   %sub_18 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_28, %mul_29), kwargs = {})
#   %mul_30 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %sub_15), kwargs = {})
#   %add_17 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %add_9), kwargs = {})
#   %gt_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_17, 0), kwargs = {})
#   %sub_16 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_15, %add_5), kwargs = {})
#   %mul_26 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %sub_15), kwargs = {})
#   %add_16 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %add_9), kwargs = {})
#   %mul_27 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %sub_15), kwargs = {})
#   %sub_17 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_27, %mul_10), kwargs = {})
#   %le_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_17, 0), kwargs = {})
#   %full_default_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default_7, %add_17), kwargs = {})
#   %div_12 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_17, %where_6), kwargs = {})
#   %where_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %div_12, %full_default_3), kwargs = {})
#   %sub_19 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_15, %where_7), kwargs = {})
#   %mul_33 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, 3.0), kwargs = {})
#   %mul_34 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 2.0), kwargs = {})
#   %sub_22 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_33, %mul_34), kwargs = {})
#   %mul_35 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %sub_19), kwargs = {})
#   %add_19 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %add_9), kwargs = {})
#   %gt_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_19, 0), kwargs = {})
#   %sub_20 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_19, %add_5), kwargs = {})
#   %mul_31 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %sub_19), kwargs = {})
#   %add_18 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %add_9), kwargs = {})
#   %mul_32 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %sub_19), kwargs = {})
#   %sub_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_32, %mul_10), kwargs = {})
#   %le_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_19, 0), kwargs = {})
#   %full_default_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %full_default_8, %add_19), kwargs = {})
#   %div_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_21, %where_8), kwargs = {})
#   %where_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %div_13, %full_default_3), kwargs = {})
#   %sub_23 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_19, %where_9), kwargs = {})
#   %mul_38 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, 3.0), kwargs = {})
#   %mul_39 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 2.0), kwargs = {})
#   %sub_26 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_38, %mul_39), kwargs = {})
#   %mul_40 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %sub_23), kwargs = {})
#   %add_21 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_40, %add_9), kwargs = {})
#   %gt_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_21, 0), kwargs = {})
#   %sub_24 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_23, %add_5), kwargs = {})
#   %mul_36 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %sub_23), kwargs = {})
#   %add_20 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %add_9), kwargs = {})
#   %mul_37 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_20, %sub_23), kwargs = {})
#   %sub_25 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_37, %mul_10), kwargs = {})
#   %le_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_21, 0), kwargs = {})
#   %full_default_9 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %full_default_9, %add_21), kwargs = {})
#   %div_14 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_25, %where_10), kwargs = {})
#   %where_11 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %div_14, %full_default_3), kwargs = {})
#   %sub_27 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_23, %where_11), kwargs = {})
#   %mul_43 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, 3.0), kwargs = {})
#   %mul_44 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 2.0), kwargs = {})
#   %sub_30 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_43, %mul_44), kwargs = {})
#   %mul_45 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %sub_27), kwargs = {})
#   %add_23 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %add_9), kwargs = {})
#   %gt_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_23, 0), kwargs = {})
#   %sub_28 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_27, %add_5), kwargs = {})
#   %mul_41 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %sub_27), kwargs = {})
#   %add_22 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %add_9), kwargs = {})
#   %mul_42 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sub_27), kwargs = {})
#   %sub_29 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_42, %mul_10), kwargs = {})
#   %le_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_23, 0), kwargs = {})
#   %full_default_10 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %full_default_10, %add_23), kwargs = {})
#   %div_15 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_29, %where_12), kwargs = {})
#   %where_13 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %div_15, %full_default_3), kwargs = {})
#   %sub_31 : Tensor "f64[32768][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_27, %where_13), kwargs = {})
#   %mul_48 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, 3.0), kwargs = {})
#   %mul_49 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 2.0), kwargs = {})
#   %sub_34 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_48, %mul_49), kwargs = {})
#   %mul_50 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %sub_31), kwargs = {})
#   %add_25 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %add_9), kwargs = {})
#   %gt_7 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_25, 0), kwargs = {})
#   %sub_32 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_31, %add_5), kwargs = {})
#   %mul_46 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %sub_31), kwargs = {})
#   %add_24 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %add_9), kwargs = {})
#   %mul_47 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sub_31), kwargs = {})
#   %sub_33 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_47, %mul_10), kwargs = {})
#   %le_7 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%add_25, 0), kwargs = {})
#   %full_default_11 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_14 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %full_default_11, %add_25), kwargs = {})
#   %div_16 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_33, %where_14), kwargs = {})
#   %where_15 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %div_16, %full_default_3), kwargs = {})
#   %sub_35 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_31, %where_15), kwargs = {})
#   %div_17 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_10, %sub_35), kwargs = {})
#   %sub_36 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %div_17), kwargs = {})
#   %div_18 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_36, %sub_35), kwargs = {})
#   %clamp_min : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_18, 0), kwargs = {})
#   %mul_51 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min, %clamp_min), kwargs = {})
#   %mul_52 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_17, 4.0), kwargs = {})
#   %sub_37 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_51, %mul_52), kwargs = {})
#   %clamp_min_1 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_37, 0), kwargs = {})
#   %sqrt_2 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_1,), kwargs = {})
#   %add_26 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min, %sqrt_2), kwargs = {})
#   %mul_53 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, 0.5), kwargs = {})
#   %mul_54 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, 3.0814879110195774e-29), kwargs = {})
#   %le_9 : Tensor "b8[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%mul_53, %mul_54), kwargs = {})
#   %gt_8 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mul_53, 0), kwargs = {})
#   %le_8 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%mul_53, 0), kwargs = {})
#   %full_default_12 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_16 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default_12, %mul_53), kwargs = {})
#   %div_19 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_17, %where_16), kwargs = {})
#   %where_17 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %div_19, %full_default_3), kwargs = {})
#   %mul_55 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, 3.0814879110195774e-29), kwargs = {})
#   %le_10 : Tensor "b8[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%where_17, %mul_55), kwargs = {})
#   %add_27 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %div_20 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_4, %add_27), kwargs = {})
#   %add_28 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_3), kwargs = {})
#   %div_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_5, %add_28), kwargs = {})
#   %le_11 : Tensor "b8[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%div_20, %div_21), kwargs = {})
#   %bitwise_and : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%le_10, %le_11), kwargs = {})
#   %bitwise_or_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_9, %bitwise_and), kwargs = {})
#   %bitwise_or_4 : Tensor "b8[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or, %bitwise_or_2), kwargs = {})
#   %full_default_16 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_18 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %full_default_16, %sum_1), kwargs = {})
#   %div_23 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_5, %where_18), kwargs = {})
#   %mul_57 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_23, %ldexp_2), kwargs = {})
#   %full_default_17 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_19 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %full_default_17, %ldexp), kwargs = {})
#   %div_24 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_57, %where_19), kwargs = {})
#   %where_20 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %full_default_15, %div_24), kwargs = {})
#   %mul_60 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_20, %add), kwargs = {})
#   %sub_42 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_2, %mul_60), kwargs = {})
#   %full_default_14 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ldexp_3 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ldexp.Tensor](args = (%full_default_14, %getitem_7), kwargs = {})
#   %unsqueeze_3 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%ldexp_3, 1), kwargs = {})
#   %div_25 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_41, %unsqueeze_3), kwargs = {})
#   %mul_58 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %div_25), kwargs = {})
#   %sum_6 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_58, [1]), kwargs = {})
#   %select_8 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_1, 1, 0), kwargs = {})
#   %select_9 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_1, 1, -1), kwargs = {})
#   %eq_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_8, %select_9), kwargs = {})
#   %mul_3 : Tensor "f64[32768, 5][5, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %sub_3), kwargs = {})
#   %sum_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [1]), kwargs = {})
#   %eq_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_4, 0), kwargs = {})
#   %bitwise_or_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%eq_2, %eq_3), kwargs = {})
#   %bitwise_not : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%le_11,), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%le_10, %bitwise_not), kwargs = {})
#   %bitwise_or_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_9, %bitwise_and_1), kwargs = {})
#   %bitwise_or_5 : Tensor "b8[32768][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or_1, %bitwise_or_3), kwargs = {})
#   %full_default_18 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %full_default_18, %sum_2), kwargs = {})
#   %div_26 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_6, %where_21), kwargs = {})
#   %mul_59 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_26, %ldexp_3), kwargs = {})
#   %full_default_19 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_22 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %full_default_19, %ldexp_1), kwargs = {})
#   %div_27 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_59, %where_22), kwargs = {})
#   %where_23 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %full_default_15, %div_27), kwargs = {})
#   %mul_61 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_23, %add_1), kwargs = {})
#   %sub_43 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_42, %mul_61), kwargs = {})
#   %full_default_21 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32768], 1), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_20 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32768], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_24 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %add, %full_default_20), kwargs = {})
#   %abs_7 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where_24,), kwargs = {})
#   %where_25 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %add_1, %full_default_20), kwargs = {})
#   %abs_8 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%where_25,), kwargs = {})
#   %maximum_4 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%abs_7, %abs_8), kwargs = {})
#   %maximum_5 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.maximum.default](args = (%full_default_21, %maximum_4), kwargs = {})
#   %div_30 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_43, %maximum_5), kwargs = {})
#   %reciprocal : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%maximum_5,), kwargs = {})
#   %mul_62 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %mul_63 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_62, %mul_62), kwargs = {})
#   %div_28 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_24, %maximum_5), kwargs = {})
#   %mul_64 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_28, %div_28), kwargs = {})
#   %add_29 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, %mul_64), kwargs = {})
#   %div_29 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_25, %maximum_5), kwargs = {})
#   %mul_65 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_29, %div_29), kwargs = {})
#   %add_30 : Tensor "f64[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %mul_65), kwargs = {})
#   return %buf6,%mean,%mean_4,%ldexp,%ldexp_2,%sum_5,%sum_1,%mean_6,%mean_1,%ldexp_1,%sum_2,%ldexp_3,%sum_6,%maximum_3,%add_5,%add_8,%le_11,%add_9,%mul_10,%div_10,%sub_11,%div_12,%where_7,%sub_19,%div_14,%where_11,%sub_27,%div_16,%where_15,%sub_35,%bitwise_or_4,%bitwise_or_3,%bitwise_or_5,%where_20,%where_23,%maximum_5,%add_30,%div_30
triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_select_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_1 = async_compile.triton('triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_select_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_out_ptr1': '*fp64', 'in_out_ptr7': '*i1', 'in_out_ptr8': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'out_ptr0': '*fp64', 'out_ptr4': '*fp64', 'out_ptr12': '*i1', 'out_ptr13': '*fp64', 'out_ptr14': '*fp64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_select_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr7', 'in_out_ptr8'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 21, 'num_store': 9, 'num_reduction': 1, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3801088, 'r0_': 6553600}}
)
@triton.jit
def triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_select_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_1(in_out_ptr0, in_out_ptr1, in_out_ptr7, in_out_ptr8, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr4, out_ptr12, out_ptr13, out_ptr14, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 25*x0), r0_mask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (50*x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (10 + 50*x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (20 + 50*x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (30 + 50*x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (40 + 50*x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (5*x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (1 + 5*x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (2 + 5*x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (3 + 5*x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (4 + 5*x0), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr3 + (5*x0), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr3 + (1 + 5*x0), None, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr3 + (2 + 5*x0), None, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr3 + (3 + 5*x0), None, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr3 + (4 + 5*x0), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr1 + (1 + 50*x0), None, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr1 + (3 + 50*x0), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr1 + (5 + 50*x0), None, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr1 + (7 + 50*x0), None, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr1 + (9 + 50*x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float64)
    tmp6 = tmp5 - tmp5
    tmp8 = tmp7 - tmp5
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10 - tmp5
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13 - tmp5
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16 - tmp5
    tmp18 = tmp15 + tmp17
    tmp19 = tl.full([1, 1], 5.0, tl.float64)
    tmp20 = (tmp18 / tmp19)
    tmp23 = tmp21 + tmp22
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = (tmp29 / tmp19)
    tmp31 = tmp6 - tmp20
    tmp32 = tl_math.abs(tmp31)
    tmp33 = tmp8 - tmp20
    tmp34 = tl_math.abs(tmp33)
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp11 - tmp20
    tmp37 = tl_math.abs(tmp36)
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = tmp14 - tmp20
    tmp40 = tl_math.abs(tmp39)
    tmp41 = triton_helpers.maximum(tmp38, tmp40)
    tmp42 = tmp17 - tmp20
    tmp43 = tl_math.abs(tmp42)
    tmp44 = triton_helpers.maximum(tmp41, tmp43)
    tmp45, tmp46 = triton_helpers.frexp(tmp44)
    tmp47 = tl.full([1, 1], 1.0, tl.float64)
    tmp48 = libdevice.ldexp(tmp47, tmp46.to(tl.int32))
    tmp49 = tmp21 - tmp30
    tmp50 = tl_math.abs(tmp49)
    tmp51 = tmp22 - tmp30
    tmp52 = tl_math.abs(tmp51)
    tmp53 = triton_helpers.maximum(tmp50, tmp52)
    tmp54 = tmp24 - tmp30
    tmp55 = tl_math.abs(tmp54)
    tmp56 = triton_helpers.maximum(tmp53, tmp55)
    tmp57 = tmp26 - tmp30
    tmp58 = tl_math.abs(tmp57)
    tmp59 = triton_helpers.maximum(tmp56, tmp58)
    tmp60 = tmp28 - tmp30
    tmp61 = tl_math.abs(tmp60)
    tmp62 = triton_helpers.maximum(tmp59, tmp61)
    tmp63, tmp64 = triton_helpers.frexp(tmp62)
    tmp65 = libdevice.ldexp(tmp47, tmp64.to(tl.int32))
    tmp66 = (tmp31 / tmp48)
    tmp67 = (tmp49 / tmp65)
    tmp68 = tmp66 * tmp67
    tmp69 = (tmp33 / tmp48)
    tmp70 = (tmp51 / tmp65)
    tmp71 = tmp69 * tmp70
    tmp72 = tmp68 + tmp71
    tmp73 = (tmp36 / tmp48)
    tmp74 = (tmp54 / tmp65)
    tmp75 = tmp73 * tmp74
    tmp76 = tmp72 + tmp75
    tmp77 = (tmp39 / tmp48)
    tmp78 = (tmp57 / tmp65)
    tmp79 = tmp77 * tmp78
    tmp80 = tmp76 + tmp79
    tmp81 = (tmp42 / tmp48)
    tmp82 = (tmp60 / tmp65)
    tmp83 = tmp81 * tmp82
    tmp84 = tmp80 + tmp83
    tmp85 = tmp66 * tmp66
    tmp86 = tmp69 * tmp69
    tmp87 = tmp85 + tmp86
    tmp88 = tmp73 * tmp73
    tmp89 = tmp87 + tmp88
    tmp90 = tmp77 * tmp77
    tmp91 = tmp89 + tmp90
    tmp92 = tmp81 * tmp81
    tmp93 = tmp91 + tmp92
    tmp96 = tmp94 + tmp95
    tmp98 = tmp96 + tmp97
    tmp100 = tmp98 + tmp99
    tmp102 = tmp100 + tmp101
    tmp103 = (tmp102 / tmp19)
    tmp105 = tmp104 - tmp104
    tmp107 = tmp106 - tmp104
    tmp108 = tmp105 + tmp107
    tmp110 = tmp109 - tmp104
    tmp111 = tmp108 + tmp110
    tmp113 = tmp112 - tmp104
    tmp114 = tmp111 + tmp113
    tmp116 = tmp115 - tmp104
    tmp117 = tmp114 + tmp116
    tmp118 = (tmp117 / tmp19)
    tmp119 = tmp105 - tmp118
    tmp120 = tl_math.abs(tmp119)
    tmp121 = tmp107 - tmp118
    tmp122 = tl_math.abs(tmp121)
    tmp123 = triton_helpers.maximum(tmp120, tmp122)
    tmp124 = tmp110 - tmp118
    tmp125 = tl_math.abs(tmp124)
    tmp126 = triton_helpers.maximum(tmp123, tmp125)
    tmp127 = tmp113 - tmp118
    tmp128 = tl_math.abs(tmp127)
    tmp129 = triton_helpers.maximum(tmp126, tmp128)
    tmp130 = tmp116 - tmp118
    tmp131 = tl_math.abs(tmp130)
    tmp132 = triton_helpers.maximum(tmp129, tmp131)
    tmp133, tmp134 = triton_helpers.frexp(tmp132)
    tmp135 = libdevice.ldexp(tmp47, tmp134.to(tl.int32))
    tmp136 = (tmp119 / tmp135)
    tmp137 = tmp136 * tmp136
    tmp138 = (tmp121 / tmp135)
    tmp139 = tmp138 * tmp138
    tmp140 = tmp137 + tmp139
    tmp141 = (tmp124 / tmp135)
    tmp142 = tmp141 * tmp141
    tmp143 = tmp140 + tmp142
    tmp144 = (tmp127 / tmp135)
    tmp145 = tmp144 * tmp144
    tmp146 = tmp143 + tmp145
    tmp147 = (tmp130 / tmp135)
    tmp148 = tmp147 * tmp147
    tmp149 = tmp146 + tmp148
    tmp150 = tmp94 - tmp103
    tmp151 = tl_math.abs(tmp150)
    tmp152 = tmp95 - tmp103
    tmp153 = tl_math.abs(tmp152)
    tmp154 = triton_helpers.maximum(tmp151, tmp153)
    tmp155 = tmp97 - tmp103
    tmp156 = tl_math.abs(tmp155)
    tmp157 = triton_helpers.maximum(tmp154, tmp156)
    tmp158 = tmp99 - tmp103
    tmp159 = tl_math.abs(tmp158)
    tmp160 = triton_helpers.maximum(tmp157, tmp159)
    tmp161 = tmp101 - tmp103
    tmp162 = tl_math.abs(tmp161)
    tmp163 = triton_helpers.maximum(tmp160, tmp162)
    tmp164, tmp165 = triton_helpers.frexp(tmp163)
    tmp166 = libdevice.ldexp(tmp47, tmp165.to(tl.int32))
    tmp167 = (tmp150 / tmp166)
    tmp168 = tmp136 * tmp167
    tmp169 = (tmp152 / tmp166)
    tmp170 = tmp138 * tmp169
    tmp171 = tmp168 + tmp170
    tmp172 = (tmp155 / tmp166)
    tmp173 = tmp141 * tmp172
    tmp174 = tmp171 + tmp173
    tmp175 = (tmp158 / tmp166)
    tmp176 = tmp144 * tmp175
    tmp177 = tmp174 + tmp176
    tmp178 = (tmp161 / tmp166)
    tmp179 = tmp147 * tmp178
    tmp180 = tmp177 + tmp179
    tmp181 = tmp5 + tmp20
    tmp182 = tl_math.abs(tmp181)
    tmp183 = tmp104 + tmp118
    tmp184 = tl_math.abs(tmp183)
    tmp185 = triton_helpers.maximum(tmp182, tmp184)
    tmp186 = triton_helpers.maximum(tmp47, tmp185)
    tmp187 = tl.full([1, 1], 0.2, tl.float64)
    tmp188 = tmp93 * tmp187
    tmp189 = libdevice.sqrt(tmp188)
    tmp190 = tmp48 * tmp189
    tmp191 = tmp149 * tmp187
    tmp192 = libdevice.sqrt(tmp191)
    tmp193 = tmp135 * tmp192
    tmp194 = triton_helpers.maximum(tmp190, tmp193)
    tmp195 = triton_helpers.maximum(tmp186, tmp194)
    tmp196 = (tmp47 / tmp195)
    tmp197 = tmp196 * tmp196
    tmp198 = (tmp181 / tmp195)
    tmp199 = tmp198 * tmp198
    tmp200 = tmp197 + tmp199
    tmp201 = (tmp183 / tmp195)
    tmp202 = tmp201 * tmp201
    tmp203 = tmp200 + tmp202
    tmp204 = (tmp190 / tmp195)
    tmp205 = tmp204 * tmp204
    tmp206 = tmp203 + tmp205
    tmp207 = (tmp193 / tmp195)
    tmp208 = tmp207 * tmp207
    tmp209 = tmp206 + tmp208
    tmp210 = tmp197 + tmp202
    tmp211 = tmp205 * tmp210
    tmp212 = tmp208 * tmp200
    tmp213 = tmp211 + tmp212
    tmp214 = (tmp205 / tmp200)
    tmp215 = (tmp208 / tmp210)
    tmp216 = tmp214 <= tmp215
    tmp217 = tmp205 * tmp208
    tmp218 = tmp213 + tmp217
    tmp219 = tmp197 * tmp205
    tmp220 = tmp219 * tmp208
    tmp221 = tl.full([1, 1], 3.0, tl.float64)
    tmp222 = tmp209 * tmp221
    tmp223 = tl.full([1, 1], 2.0, tl.float64)
    tmp224 = tmp209 * tmp223
    tmp225 = tmp222 - tmp224
    tmp226 = tmp225 * tmp209
    tmp227 = tmp226 + tmp218
    tmp228 = tl.full([1, 1], 0.0, tl.float64)
    tmp229 = tmp227 > tmp228
    tmp230 = tmp209 - tmp209
    tmp231 = tmp230 * tmp209
    tmp232 = tmp231 + tmp218
    tmp233 = tmp232 * tmp209
    tmp234 = tmp233 - tmp220
    tmp235 = tmp227 <= tmp228
    tmp236 = tl.where(tmp235, tmp47, tmp227)
    tmp237 = (tmp234 / tmp236)
    tmp238 = tl.where(tmp229, tmp237, tmp228)
    tmp239 = tmp209 - tmp238
    tmp240 = tmp239 - tmp209
    tmp241 = tmp240 * tmp239
    tmp242 = tmp241 + tmp218
    tmp243 = tmp242 * tmp239
    tmp244 = tmp243 - tmp220
    tmp245 = tmp239 * tmp221
    tmp246 = tmp245 - tmp224
    tmp247 = tmp246 * tmp239
    tmp248 = tmp247 + tmp218
    tmp249 = tmp248 <= tmp228
    tmp250 = tl.where(tmp249, tmp47, tmp248)
    tmp251 = (tmp244 / tmp250)
    tmp252 = tmp248 > tmp228
    tmp253 = tl.where(tmp252, tmp251, tmp228)
    tmp254 = tmp239 - tmp253
    tmp255 = tmp254 * tmp221
    tmp256 = tmp255 - tmp224
    tmp257 = tmp256 * tmp254
    tmp258 = tmp257 + tmp218
    tmp259 = tmp258 > tmp228
    tmp260 = tmp254 - tmp209
    tmp261 = tmp260 * tmp254
    tmp262 = tmp261 + tmp218
    tmp263 = tmp262 * tmp254
    tmp264 = tmp263 - tmp220
    tmp265 = tmp258 <= tmp228
    tmp266 = tl.where(tmp265, tmp47, tmp258)
    tmp267 = (tmp264 / tmp266)
    tmp268 = tl.where(tmp259, tmp267, tmp228)
    tmp269 = tmp254 - tmp268
    tmp270 = tmp269 - tmp209
    tmp271 = tmp270 * tmp269
    tmp272 = tmp271 + tmp218
    tmp273 = tmp272 * tmp269
    tmp274 = tmp273 - tmp220
    tmp275 = tmp269 * tmp221
    tmp276 = tmp275 - tmp224
    tmp277 = tmp276 * tmp269
    tmp278 = tmp277 + tmp218
    tmp279 = tmp278 <= tmp228
    tmp280 = tl.where(tmp279, tmp47, tmp278)
    tmp281 = (tmp274 / tmp280)
    tmp282 = tmp278 > tmp228
    tmp283 = tl.where(tmp282, tmp281, tmp228)
    tmp284 = tmp269 - tmp283
    tmp285 = tmp284 * tmp221
    tmp286 = tmp285 - tmp224
    tmp287 = tmp286 * tmp284
    tmp288 = tmp287 + tmp218
    tmp289 = tmp288 > tmp228
    tmp290 = tmp284 - tmp209
    tmp291 = tmp290 * tmp284
    tmp292 = tmp291 + tmp218
    tmp293 = tmp292 * tmp284
    tmp294 = tmp293 - tmp220
    tmp295 = tmp288 <= tmp228
    tmp296 = tl.where(tmp295, tmp47, tmp288)
    tmp297 = (tmp294 / tmp296)
    tmp298 = tl.where(tmp289, tmp297, tmp228)
    tmp299 = tmp284 - tmp298
    tmp300 = tmp299 - tmp209
    tmp301 = tmp300 * tmp299
    tmp302 = tmp301 + tmp218
    tmp303 = tmp302 * tmp299
    tmp304 = tmp303 - tmp220
    tmp305 = tmp299 * tmp221
    tmp306 = tmp305 - tmp224
    tmp307 = tmp306 * tmp299
    tmp308 = tmp307 + tmp218
    tmp309 = tmp308 <= tmp228
    tmp310 = tl.where(tmp309, tmp47, tmp308)
    tmp311 = (tmp304 / tmp310)
    tmp312 = tmp308 > tmp228
    tmp313 = tl.where(tmp312, tmp311, tmp228)
    tmp314 = tmp299 - tmp313
    tmp315 = tmp314 * tmp221
    tmp316 = tmp315 - tmp224
    tmp317 = tmp316 * tmp314
    tmp318 = tmp317 + tmp218
    tmp319 = tmp318 > tmp228
    tmp320 = tmp314 - tmp209
    tmp321 = tmp320 * tmp314
    tmp322 = tmp321 + tmp218
    tmp323 = tmp322 * tmp314
    tmp324 = tmp323 - tmp220
    tmp325 = tmp318 <= tmp228
    tmp326 = tl.where(tmp325, tmp47, tmp318)
    tmp327 = (tmp324 / tmp326)
    tmp328 = tl.where(tmp319, tmp327, tmp228)
    tmp329 = tmp314 - tmp328
    tmp330 = tmp329 - tmp209
    tmp331 = tmp330 * tmp329
    tmp332 = tmp331 + tmp218
    tmp333 = tmp332 * tmp329
    tmp334 = tmp333 - tmp220
    tmp335 = tmp329 * tmp221
    tmp336 = tmp335 - tmp224
    tmp337 = tmp336 * tmp329
    tmp338 = tmp337 + tmp218
    tmp339 = tmp338 <= tmp228
    tmp340 = tl.where(tmp339, tmp47, tmp338)
    tmp341 = (tmp334 / tmp340)
    tmp342 = tmp338 > tmp228
    tmp343 = tl.where(tmp342, tmp341, tmp228)
    tmp344 = tmp329 - tmp343
    tmp345 = tmp5 == tmp16
    tmp346 = tmp31 * tmp31
    tmp347 = tmp33 * tmp33
    tmp348 = tmp346 + tmp347
    tmp349 = tmp36 * tmp36
    tmp350 = tmp348 + tmp349
    tmp351 = tmp39 * tmp39
    tmp352 = tmp350 + tmp351
    tmp353 = tmp42 * tmp42
    tmp354 = tmp352 + tmp353
    tmp355 = tmp354 == tmp228
    tmp356 = tmp345 | tmp355
    tmp357 = (tmp220 / tmp344)
    tmp358 = tmp218 - tmp357
    tmp359 = (tmp358 / tmp344)
    tmp360 = triton_helpers.maximum(tmp359, tmp228)
    tmp361 = tmp360 * tmp360
    tmp362 = tl.full([1, 1], 4.0, tl.float64)
    tmp363 = tmp357 * tmp362
    tmp364 = tmp361 - tmp363
    tmp365 = triton_helpers.maximum(tmp364, tmp228)
    tmp366 = libdevice.sqrt(tmp365)
    tmp367 = tmp360 + tmp366
    tmp368 = tl.full([1, 1], 0.5, tl.float64)
    tmp369 = tmp367 * tmp368
    tmp370 = tl.full([1, 1], 3.0814879110195774e-29, tl.float64)
    tmp371 = tmp344 * tmp370
    tmp372 = tmp369 <= tmp371
    tmp373 = tmp369 > tmp228
    tmp374 = tmp369 <= tmp228
    tmp375 = tl.where(tmp374, tmp47, tmp369)
    tmp376 = (tmp357 / tmp375)
    tmp377 = tl.where(tmp373, tmp376, tmp228)
    tmp378 = tmp377 <= tmp371
    tmp379 = tmp378 & tmp216
    tmp380 = tmp372 | tmp379
    tmp381 = tmp356 | tmp380
    tmp382 = tmp216 == 0
    tmp383 = tmp378 & tmp382
    tmp384 = tmp372 | tmp383
    tmp385 = tmp104 == tmp115
    tmp386 = tmp119 * tmp119
    tmp387 = tmp121 * tmp121
    tmp388 = tmp386 + tmp387
    tmp389 = tmp124 * tmp124
    tmp390 = tmp388 + tmp389
    tmp391 = tmp127 * tmp127
    tmp392 = tmp390 + tmp391
    tmp393 = tmp130 * tmp130
    tmp394 = tmp392 + tmp393
    tmp395 = tmp394 == tmp228
    tmp396 = tmp385 | tmp395
    tmp397 = tmp396 | tmp384
    tmp398 = tl.where(tmp381, tmp47, tmp93)
    tmp399 = (tmp84 / tmp398)
    tmp400 = tmp399 * tmp65
    tmp401 = tl.where(tmp381, tmp47, tmp48)
    tmp402 = (tmp400 / tmp401)
    tmp403 = tl.where(tmp381, tmp228, tmp402)
    tmp404 = tl.where(tmp397, tmp47, tmp149)
    tmp405 = (tmp180 / tmp404)
    tmp406 = tmp405 * tmp166
    tmp407 = tl.where(tmp397, tmp47, tmp135)
    tmp408 = (tmp406 / tmp407)
    tmp409 = tl.where(tmp397, tmp228, tmp408)
    tmp410 = tl.where(tmp381, tmp181, tmp228)
    tmp411 = tl_math.abs(tmp410)
    tmp412 = tl.where(tmp397, tmp183, tmp228)
    tmp413 = tl_math.abs(tmp412)
    tmp414 = triton_helpers.maximum(tmp411, tmp413)
    tmp415 = triton_helpers.maximum(tmp47, tmp414)
    tmp416 = tl.full([1, 1], 1, tl.int32)
    tmp417 = (tmp416 / tmp415)
    tmp418 = tmp417 * tmp47
    tmp419 = tmp418 * tmp418
    tmp420 = (tmp410 / tmp415)
    tmp421 = tmp420 * tmp420
    tmp422 = tmp419 + tmp421
    tmp423 = (tmp412 / tmp415)
    tmp424 = tmp423 * tmp423
    tmp425 = tmp422 + tmp424
    tmp426 = tl.full([1, 1], 25.0, tl.float64)
    tmp427 = (tmp4 / tmp426)
    tmp428 = tmp403 * tmp181
    tmp429 = tmp427 - tmp428
    tmp430 = tmp409 * tmp183
    tmp431 = tmp429 - tmp430
    tmp432 = (tmp431 / tmp415)
    tl.store(out_ptr0 + (x0), tmp20, None)
    tl.store(out_ptr4 + (x0), tmp118, None)
    tl.store(out_ptr12 + (x0), tmp381, None)
    tl.debug_barrier()
    tl.store(in_out_ptr7 + (x0), tmp397, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp403, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp409, None)
    tl.store(out_ptr13 + (x0), tmp415, None)
    tl.store(out_ptr14 + (x0), tmp425, None)
    tl.debug_barrier()
    tl.store(in_out_ptr8 + (x0), tmp432, None)
''', device_str='cuda')


# kernel path: /content/inductor_caches/logs_big_S32_batched_A2_compiled_fit/inductor/gh/cghfslj25wafpocds3gzgeewnm7m5yefnzrb345zpvzqwp6brzwl.py
# Topologically Sorted Source Nodes: [xs, ys, getitem_3, getitem_4, xbar, getitem_6, getitem_7, ybar, zero_2, u, v, inv, mul_65, un, vn, C, stack, mul_66, truediv_33, A, mul_67, truediv_34, B], Original ATen: [aten.slice, aten.select, aten.add, aten.zeros_like, aten.where, aten.reciprocal, aten.mul, aten.div, aten.stack]
# Source node to ATen node mapping:
#   A => where_26
#   B => where_27
#   C => div_31
#   getitem_3 => select_2
#   getitem_4 => select_3
#   getitem_6 => select_4
#   getitem_7 => select_5
#   inv => mul_62, reciprocal
#   mul_65 => mul_66
#   mul_66 => mul_67
#   mul_67 => mul_68
#   stack => cat, unsqueeze_4, unsqueeze_5, unsqueeze_6
#   truediv_33 => div_32
#   truediv_34 => div_33
#   u => where_24
#   un => div_28
#   v => where_25
#   vn => div_29
#   xbar => add
#   xs => select, slice_1
#   ybar => add_1
#   ys => select_1, slice_2
#   zero_2 => full_default_20
# Graph fragment:
#   %div_30 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=div_30]
#   %maximum_5 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=maximum_5]
#   %add_30 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=add_30]
#   %bitwise_or_4 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_4]
#   %arg1_1 : Tensor "f64[32768, 25, 2][50, 2, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mean : Tensor "f64[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=mean]
#   %where_20 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_20]
#   %bitwise_or_5 : Tensor "b8[32768][1]cuda:0" = PlaceHolder[target=bitwise_or_5]
#   %mean_1 : Tensor "f64[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=mean_1]
#   %where_23 : Tensor "f64[32768][1]cuda:0" = PlaceHolder[target=where_23]
#   %slice_1 : Tensor "f64[32768, 5, 2][50, 10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 0, 9223372036854775807, 5), kwargs = {})
#   %select : Tensor "f64[32768, 5][50, 10]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%slice_1, 2, 0), kwargs = {})
#   %slice_2 : Tensor "f64[32768, 5, 2][50, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 0, 5), kwargs = {})
#   %select_1 : Tensor "f64[32768, 5][50, 2]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%slice_2, 2, 1), kwargs = {})
#   %select_2 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select, 1, 0), kwargs = {})
#   %select_3 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%mean, 1, 0), kwargs = {})
#   %add : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, %select_3), kwargs = {})
#   %select_4 : Tensor "f64[32768][50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_1, 1, 0), kwargs = {})
#   %select_5 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%mean_1, 1, 0), kwargs = {})
#   %add_1 : Tensor "f64[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_4, %select_5), kwargs = {})
#   %full_default_20 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32768], 0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_24 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %add, %full_default_20), kwargs = {})
#   %where_25 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %add_1, %full_default_20), kwargs = {})
#   %reciprocal : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%maximum_5,), kwargs = {})
#   %mul_62 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %mul_66 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_30, %mul_62), kwargs = {})
#   %div_28 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_24, %maximum_5), kwargs = {})
#   %div_29 : Tensor "f64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_25, %maximum_5), kwargs = {})
#   %div_31 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_66, %add_30), kwargs = {})
#   %unsqueeze_4 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%div_31, 1), kwargs = {})
#   %mul_67 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_30, %div_28), kwargs = {})
#   %div_32 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_67, %add_30), kwargs = {})
#   %where_26 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_4, %div_32, %where_20), kwargs = {})
#   %unsqueeze_5 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where_26, 1), kwargs = {})
#   %mul_68 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_30, %div_29), kwargs = {})
#   %div_33 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_68, %add_30), kwargs = {})
#   %where_27 : Tensor "f64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or_5, %div_33, %where_23), kwargs = {})
#   %unsqueeze_6 : Tensor "f64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where_27, 1), kwargs = {})
#   %cat : Tensor "f64[32768, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_4, %unsqueeze_5, %unsqueeze_6], 1), kwargs = {})
#   return %cat
triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_2 = async_compile.triton('triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_2', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 17, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 1900544, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (32768, 25), (25, 1))
        assert_size_stride(arg1_1, (32768, 25, 2), (50, 2, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = empty_strided_cuda((32768, 5), (5, 1), torch.float64)
            buf4 = empty_strided_cuda((32768, 5), (5, 1), torch.float64)
            # Topologically Sorted Source Nodes: [z3, getitem_18, sub_38, px, getitem_19, sub_40, py], Original ATen: [aten.view, aten.slice, aten.sub, aten.mean]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mean_slice_sub_view_0.run(arg0_1, buf2, buf4, 163840, stream=stream0)
            buf6 = empty_strided_cuda((32768, ), (1, ), torch.float64)
            buf0 = empty_strided_cuda((32768, 1), (1, 32768), torch.float64)
            buf3 = empty_strided_cuda((32768, 1), (1, 32768), torch.float64)
            buf9 = reinterpret_tensor(buf3, (32768, ), (1, ), 0); del buf3  # reuse
            buf5 = empty_strided_cuda((32768, 1), (1, 32768), torch.float64)
            buf1 = empty_strided_cuda((32768, 1), (1, 32768), torch.float64)
            buf33 = reinterpret_tensor(buf5, (32768, ), (1, ), 0); del buf5  # reuse
            buf30 = empty_strided_cuda((32768, ), (1, ), torch.bool)
            buf34 = empty_strided_cuda((32768, ), (1, ), torch.bool)
            buf35 = buf34; del buf34  # reuse
            buf31 = buf9; del buf9  # reuse
            buf36 = buf33; del buf33  # reuse
            buf37 = empty_strided_cuda((32768, ), (1, ), torch.float64)
            buf39 = empty_strided_cuda((32768, ), (1, ), torch.float64)
            buf38 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [xs, getitem_2, r, rm, dx, abs_1, m, frexp, ys, getitem_5, r_1, rm_1, dy, abs_2, m_1, frexp_1, mean_4, px_1, abs_5, m_3, frexp_2, mean_6, py_1, abs_6, m_4, frexp_3, zbar, zero_1, ones_like, ax, getitem_12, dxn, ones_like_12, sx, getitem_24, truediv_22, mul_56, sum_5, getitem_14, getitem_15, eq, mul_2, sum_3, eq_1, xdeg, one, getitem_3, getitem_4, xbar, abs_3, getitem_6, getitem_7, ybar, abs_4, maximum, maximum_1, mul, nxx, truediv_2, sqrt, mul_4, ones_like_1, ay, getitem_13, dyn, mul_1, nyy, truediv_3, sqrt_1, mul_5, maximum_2, m_2, truediv_4, c2, truediv_5, a2, add_2, truediv_6, b2, add_3, truediv_7, p2, add_4, truediv_8, q2, e1, mul_13, mul_14, sub_6, mul_15, add_6, mul_6, add_7, mul_7, add_8, mul_8, e2, df, gt, sub_4, mul_11, add_10, mul_12, mul_9, e3, f, le, ones_like_3, where, truediv_9, zero, where_1, l1, mul_18, mul_19, sub_10, mul_20, df_1, gt_1, sub_8, mul_16, add_12, mul_17, f_1, le_1, ones_like_4, where_2, truediv_10, where_3, l1_1, mul_23, mul_24, sub_14, mul_25, df_2, gt_2, sub_12, mul_21, add_14, mul_22, f_2, le_2, ones_like_5, where_4, truediv_11, where_5, l1_2, mul_28, mul_29, sub_18, mul_30, df_3, gt_3, sub_16, mul_26, add_16, mul_27, f_3, le_3, ones_like_6, where_6, truediv_12, where_7, l1_3, mul_33, mul_34, sub_22, mul_35, df_4, gt_4, sub_20, mul_31, add_18, mul_32, f_4, le_4, ones_like_7, where_8, truediv_13, where_9, l1_4, mul_38, mul_39, sub_26, mul_40, df_5, gt_5, sub_24, mul_36, add_20, mul_37, f_5, le_5, ones_like_8, where_10, truediv_14, where_11, l1_5, mul_43, mul_44, sub_30, mul_45, df_6, gt_6, sub_28, mul_41, add_22, mul_42, f_6, le_6, ones_like_9, where_12, truediv_15, where_13, l1_6, mul_48, mul_49, sub_34, mul_50, df_7, gt_7, sub_32, mul_46, add_24, mul_47, f_7, le_7, ones_like_10, where_14, truediv_16, where_15, l1_7, prod, sub_36, truediv_18, total, mul_51, mul_52, sub_37, disc, sqrt_2, add_26, l2, mul_54, drop2, gt_8, le_8, ones_like_11, where_16, truediv_19, l3, mul_55, drop3, add_27, tx, add_28, ty, pick_x, and_, cx, or__4, ones_like_14, where_18, truediv_23, mul_57, ones_like_15, where_19, truediv_24, a0, mul_60, sub_42, ones_like_13, sy, getitem_25, truediv_25, mul_58, sum_6, getitem_16, getitem_17, eq_2, mul_3, sum_4, eq_3, ydeg, invert, and__1, cy, or__5, ones_like_16, where_21, truediv_26, mul_59, ones_like_17, where_22, truediv_27, b0, mul_61, k, ones_like_18, zero_2, u, abs_7, v, abs_8, maximum_4, m_5, km, inv, mul_62, un, mul_63, add_29, vn, mul_64, denom], Original ATen: [aten.slice, aten.select, aten.sub, aten.mean, aten.abs, aten.amax, aten.frexp, aten.zeros_like, aten.ones_like, aten.ldexp, aten.unsqueeze, aten.div, aten.mul, aten.sum, aten.eq, aten.bitwise_or, aten.add, aten.maximum, aten.sqrt, aten.pow, aten.gt, aten.le, aten.where, aten.clamp_min, aten.bitwise_and, aten.bitwise_not, aten.reciprocal]
            stream0 = get_raw_stream(0)
            triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_select_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_1.run(buf31, buf36, buf35, buf38, arg0_1, arg1_1, buf2, buf4, buf0, buf1, buf30, buf37, buf39, 32768, 25, stream=stream0)
            del arg0_1
            del buf2
            del buf4
            buf40 = empty_strided_cuda((32768, 3), (3, 1), torch.float64)
            # Topologically Sorted Source Nodes: [xs, ys, getitem_3, getitem_4, xbar, getitem_6, getitem_7, ybar, zero_2, u, v, inv, mul_65, un, vn, C, stack, mul_66, truediv_33, A, mul_67, truediv_34, B], Original ATen: [aten.slice, aten.select, aten.add, aten.zeros_like, aten.where, aten.reciprocal, aten.mul, aten.div, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_mul_reciprocal_select_slice_stack_where_zeros_like_2.run(buf38, buf37, buf39, buf30, arg1_1, buf0, buf31, buf35, buf1, buf36, buf40, 32768, 3, stream=stream0)
            del arg1_1
            del buf0
            del buf1
            del buf30
            del buf31
            del buf35
            del buf36
            del buf37
            del buf38
            del buf39
        return (buf40, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((32768, 25), (25, 1), device='cuda:0', dtype=torch.float64)
    arg1_1 = rand_strided((32768, 25, 2), (50, 2, 1), device='cuda:0', dtype=torch.float64)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
