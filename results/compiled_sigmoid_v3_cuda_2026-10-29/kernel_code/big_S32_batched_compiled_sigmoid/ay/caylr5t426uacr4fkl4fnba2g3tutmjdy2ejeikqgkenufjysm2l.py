
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
