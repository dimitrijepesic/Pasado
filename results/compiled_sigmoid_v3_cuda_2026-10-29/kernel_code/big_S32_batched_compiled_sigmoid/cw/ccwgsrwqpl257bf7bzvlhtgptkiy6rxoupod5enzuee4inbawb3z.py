
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
