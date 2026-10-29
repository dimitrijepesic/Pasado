
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr2': '*i1', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'in_ptr4': '*fp64', 'in_ptr5': '*fp64', 'in_ptr6': '*fp64', 'in_ptr7': '*fp64', 'out_ptr0': '*fp64', 'out_ptr7': '*fp64', 'out_ptr9': '*fp64', 'out_ptr10': '*fp64', 'out_ptr11': '*fp64', 'out_ptr12': '*fp64', 'out_ptr13': '*fp64', 'out_ptr14': '*fp64', 'out_ptr15': '*fp64', 'ks0': 'i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_acos_add_all_bitwise_or_clamp_copy_cos_diagonal_div_full_like_ge_gt_index_put_isinf_le_lift_fresh_log_logical_and_logical_not_lt_minimum_mul_neg_rsub_select_sigmoid_slice_stack_sub_view_where_8', 'mutated_arg_names': ['in_out_ptr2', 'out_ptr15'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 10, 'num_reduction': 1, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 18400}}
)
@triton.jit
def triton_red_fused_abs_acos_add_all_bitwise_or_clamp_copy_cos_diagonal_div_full_like_ge_gt_index_put_isinf_le_lift_fresh_log_logical_and_logical_not_lt_minimum_mul_neg_rsub_select_sigmoid_slice_stack_sub_view_where_8(in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr7, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 100
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp177 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp16 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr1 + (2 + 3*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr5 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_ptr1 + (3*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp69 = tl.load(in_ptr6 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp71 = tl.load(in_ptr7 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = tl.full([1, 1], 0, tl.int64)
        tmp1 = tmp0 >= tmp0
        tmp2 = tl.full([1, 1], 1, tl.int64)
        tmp3 = tmp0 < tmp2
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), r0_mask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp0 >= tmp2
        tmp6 = tl.full([1, 1], 2, tl.int64)
        tmp7 = tmp0 < tmp6
        tmp8 = tmp5 & tmp7
        tmp9 = tl.load(in_ptr1 + (tl.broadcast_to(2 + 3*r0_0, [XBLOCK, R0_BLOCK])), r0_mask & tmp8, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp0 >= tmp6
        tmp11 = tl.full([1, 1], 3, tl.int64)
        tmp12 = tmp0 < tmp11
        tmp13 = tl.load(in_ptr1 + (tl.broadcast_to(3*r0_0, [XBLOCK, R0_BLOCK])), r0_mask & tmp10, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.where(tmp8, tmp9, tmp13)
        tmp15 = tl.where(tmp3, tmp4, tmp14)
        tmp17 = (tmp15 / tmp16)
        tmp18 = libdevice.isinf(tmp17).to(tl.int1)
        tmp19 = tl.full([1, 1], float("nan"), tl.float64)
        tmp20 = tl.where(tmp18, tmp19, tmp17)
        tmp22 = (tmp15 / tmp21)
        tmp23 = libdevice.isinf(tmp22).to(tl.int1)
        tmp24 = tl.where(tmp23, tmp19, tmp22)
        tmp27 = tmp25 * tmp26
        tmp30 = tmp28 * tmp29
        tmp31 = tmp27 + tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = tl.full([1, 1], 0.09622504486493763, tl.float64)
        tmp35 = tmp20 <= tmp34
        tmp36 = tl.full([1, 1], -0.09622504486493763, tl.float64)
        tmp37 = tmp20 >= tmp36
        tmp38 = tmp35 & tmp37
        tmp39 = tmp38 == 0
        tmp40 = tl.where(tmp39, tmp19, tmp20)
        tmp41 = tmp24 <= tmp34
        tmp42 = tmp24 >= tmp36
        tmp43 = tmp41 & tmp42
        tmp44 = tmp43 == 0
        tmp45 = tl.where(tmp44, tmp19, tmp24)
        tmp46 = tl.full([1, 1], 10.392304845413264, tl.float64)
        tmp47 = tmp40 * tmp46
        tmp48 = tl_math.abs(tmp47)
        tmp49 = tl.full([1, 1], 1.0000001, tl.float64)
        tmp50 = tmp48 <= tmp49
        tmp51 = tl.full([1, 1], -1.0, tl.float64)
        tmp52 = triton_helpers.maximum(tmp47, tmp51)
        tmp53 = tl.full([1, 1], 1.0, tl.float64)
        tmp54 = triton_helpers.minimum(tmp52, tmp53)
        tmp55 = libdevice.acos(tmp54)
        tmp56 = tl.full([1, 1], 0.3333333333333333, tl.float64)
        tmp57 = tmp55 * tmp56
        tmp58 = tl.where(tmp50, tmp57, tmp19)
        tmp59 = tl.full([1, 1], 0.0, tl.float64)
        tmp60 = tmp58 - tmp59
        tmp61 = libdevice.cos(tmp60)
        tmp62 = tl.full([1, 1], 0.5773502691896258, tl.float64)
        tmp63 = tmp61 * tmp62
        tmp64 = tl.full([1, 1], 0.5, tl.float64)
        tmp65 = tmp63 + tmp64
        tmp66 = tmp53 - tmp65
        tmp67 = (tmp65 / tmp66)
        tmp68 = libdevice.log(tmp67)
        tmp70 = tmp68 <= tmp69
        tmp72 = tmp68 >= tmp71
        tmp73 = tmp70 & tmp72
        tmp74 = tmp73 == 0
        tmp75 = tl.full([1, 1], 2.0943951023931953, tl.float64)
        tmp76 = tmp58 - tmp75
        tmp77 = libdevice.cos(tmp76)
        tmp78 = tmp77 * tmp62
        tmp79 = tmp78 + tmp64
        tmp80 = tmp53 - tmp79
        tmp81 = (tmp79 / tmp80)
        tmp82 = libdevice.log(tmp81)
        tmp83 = tmp82 <= tmp69
        tmp84 = tmp82 >= tmp71
        tmp85 = tmp83 & tmp84
        tmp86 = tmp85 == 0
        tmp87 = tl.full([1, 1], 4.1887902047863905, tl.float64)
        tmp88 = tmp58 - tmp87
        tmp89 = libdevice.cos(tmp88)
        tmp90 = tmp89 * tmp62
        tmp91 = tmp90 + tmp64
        tmp92 = tmp53 - tmp91
        tmp93 = (tmp91 / tmp92)
        tmp94 = libdevice.log(tmp93)
        tmp95 = tmp94 <= tmp69
        tmp96 = tmp94 >= tmp71
        tmp97 = tmp95 & tmp96
        tmp98 = tmp97 == 0
        tmp99 = tmp45 * tmp46
        tmp100 = tl_math.abs(tmp99)
        tmp101 = tmp100 <= tmp49
        tmp102 = triton_helpers.maximum(tmp99, tmp51)
        tmp103 = triton_helpers.minimum(tmp102, tmp53)
        tmp104 = libdevice.acos(tmp103)
        tmp105 = tmp104 * tmp56
        tmp106 = tl.where(tmp101, tmp105, tmp19)
        tmp107 = tmp106 - tmp59
        tmp108 = libdevice.cos(tmp107)
        tmp109 = tmp108 * tmp62
        tmp110 = tmp109 + tmp64
        tmp111 = tmp53 - tmp110
        tmp112 = (tmp110 / tmp111)
        tmp113 = libdevice.log(tmp112)
        tmp114 = tmp113 <= tmp69
        tmp115 = tmp113 >= tmp71
        tmp116 = tmp114 & tmp115
        tmp117 = tmp116 == 0
        tmp118 = tmp106 - tmp75
        tmp119 = libdevice.cos(tmp118)
        tmp120 = tmp119 * tmp62
        tmp121 = tmp120 + tmp64
        tmp122 = tmp53 - tmp121
        tmp123 = (tmp121 / tmp122)
        tmp124 = libdevice.log(tmp123)
        tmp125 = tmp124 <= tmp69
        tmp126 = tmp124 >= tmp71
        tmp127 = tmp125 & tmp126
        tmp128 = tmp127 == 0
        tmp129 = tmp106 - tmp87
        tmp130 = libdevice.cos(tmp129)
        tmp131 = tmp130 * tmp62
        tmp132 = tmp131 + tmp64
        tmp133 = tmp53 - tmp132
        tmp134 = (tmp132 / tmp133)
        tmp135 = libdevice.log(tmp134)
        tmp136 = tmp135 <= tmp69
        tmp137 = tmp135 >= tmp71
        tmp138 = tmp136 & tmp137
        tmp139 = tmp138 == 0
        tmp140 = tl.full([1, 1], 1, tl.int32)
        tmp141 = -tmp71
        tmp142 = libdevice.exp(tmp141)
        tmp143 = tmp140 + tmp142
        tmp144 = (tmp140 / tmp143)
        tmp145 = tmp53 - tmp144
        tmp146 = tmp144 * tmp145
        tmp147 = -tmp69
        tmp148 = libdevice.exp(tmp147)
        tmp149 = tmp140 + tmp148
        tmp150 = (tmp140 / tmp149)
        tmp151 = tmp53 - tmp150
        tmp152 = tmp150 * tmp151
        tmp153 = triton_helpers.minimum(tmp146, tmp152)
        tmp154 = tmp153 * tmp26
        tmp155 = tmp150 + tmp144
        tmp156 = tmp69 + tmp71
        tmp157 = tmp153 * tmp156
        tmp158 = tmp155 - tmp157
        tmp159 = tmp158 * tmp64
        tmp160 = tmp154 + tmp159
        tmp161 = tmp150 - tmp144
        tmp162 = tmp69 - tmp71
        tmp163 = tmp153 * tmp162
        tmp164 = tmp161 - tmp163
        tmp165 = tmp164 * tmp64
        tmp166 = tl.where(tmp74, tmp19, tmp68)
        tmp167 = tl.where(tmp86, tmp19, tmp82)
        tmp168 = tl.where(tmp98, tmp19, tmp94)
        tmp169 = tl.where(tmp117, tmp19, tmp113)
        tmp170 = tl.where(tmp128, tmp19, tmp124)
        tmp171 = tl.where(tmp139, tmp19, tmp135)
        tmp172 = tmp25 > tmp59
        tmp173 = tmp25 < tmp59
        tmp174 = tmp172 | tmp173
        tmp175 = tmp174 == 0
        tmp176 = tl.broadcast_to(tmp175, [XBLOCK, R0_BLOCK])
        tmp178 = _tmp177 | tmp176
        _tmp177 = tl.where(r0_mask, tmp178, _tmp177)
        tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp33, r0_mask)
        tl.store(out_ptr7 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp160, r0_mask)
        tl.store(out_ptr9 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp166, r0_mask)
        tl.store(out_ptr10 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp167, r0_mask)
        tl.store(out_ptr11 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp168, r0_mask)
        tl.store(out_ptr12 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp169, r0_mask)
        tl.store(out_ptr13 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp170, r0_mask)
        tl.store(out_ptr14 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp171, r0_mask)
        tl.store(out_ptr15 + (tl.broadcast_to(100*ks0 + 101*r0_0, [XBLOCK, R0_BLOCK])), tmp165, r0_mask)
    tmp179 = _tmp177.to(tl.int8)
    tmp177 = triton_helpers.any(tmp179, 1)[:, None]
    tmp180 = tmp177 == 0
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp180, None)
