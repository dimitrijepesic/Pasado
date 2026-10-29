
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
