
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_out_ptr1': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'in_ptr4': '*fp64', 'in_ptr5': '*fp64', 'in_ptr6': '*fp64', 'in_ptr7': '*fp64', 'in_ptr8': '*fp64', 'in_ptr9': '*fp64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 17, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 7200, 'x': 3600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_mul_rsub_select_sigmoid_stack_sub_unsqueeze_view_9(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp42 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp3 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp0 >= tmp2
    tmp6 = tl.full([1, 1], 2, tl.int64)
    tmp7 = tmp0 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tl.load(in_ptr1 + (tl.broadcast_to(2 + 3*y0, [YBLOCK, XBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp0 >= tmp6
    tmp11 = tl.full([1, 1], 3, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr1 + (tl.broadcast_to(3*y0, [YBLOCK, XBLOCK])), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp8, tmp9, tmp13)
    tmp15 = tl.where(tmp3, tmp4, tmp14)
    tmp16 = x1
    tmp17 = tmp16 >= tmp0
    tmp18 = tmp16 < tmp2
    tmp19 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp16 >= tmp2
    tmp21 = tmp16 < tmp6
    tmp22 = tmp20 & tmp21
    tmp23 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp16 >= tmp6
    tmp25 = tmp16 < tmp11
    tmp26 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp22, tmp23, tmp26)
    tmp28 = tl.where(tmp18, tmp19, tmp27)
    tmp29 = tmp15 * tmp28
    tmp30 = tmp2 >= tmp0
    tmp31 = tmp2 < tmp2
    tmp32 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp31 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp2 >= tmp2
    tmp34 = tmp2 < tmp6
    tmp35 = tmp33 & tmp34
    tmp36 = tl.load(in_ptr1 + (tl.broadcast_to(2 + 3*y0, [YBLOCK, XBLOCK])), tmp35 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp2 >= tmp6
    tmp38 = tmp2 < tmp11
    tmp39 = tl.load(in_ptr1 + (tl.broadcast_to(3*y0, [YBLOCK, XBLOCK])), tmp37 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp35, tmp36, tmp39)
    tmp41 = tl.where(tmp31, tmp32, tmp40)
    tmp43 = tmp41 * tmp42
    tmp44 = tmp29 + tmp43
    tmp45 = tmp6 >= tmp0
    tmp46 = tmp6 < tmp2
    tmp47 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp6 >= tmp2
    tmp49 = tmp6 < tmp6
    tmp50 = tmp48 & tmp49
    tmp51 = tl.load(in_ptr1 + (tl.broadcast_to(2 + 3*y0, [YBLOCK, XBLOCK])), tmp50 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp6 >= tmp6
    tmp53 = tmp6 < tmp11
    tmp54 = tl.load(in_ptr1 + (tl.broadcast_to(3*y0, [YBLOCK, XBLOCK])), tmp52 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tmp69 = tl.load(in_ptr6 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp70 = tl.load(in_ptr7 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.load(in_ptr8 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tl.store(in_out_ptr0 + (x1 + 3*y0), tmp68, xmask & ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x1 + 3*y0), tmp87, xmask & ymask)
