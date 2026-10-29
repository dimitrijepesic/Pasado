
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
