
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
