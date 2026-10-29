
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_out_ptr1': '*fp64', 'in_out_ptr7': '*i1', 'in_out_ptr8': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'out_ptr0': '*fp64', 'out_ptr4': '*fp64', 'out_ptr12': '*i1', 'out_ptr13': '*fp64', 'out_ptr14': '*fp64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_rsub_select_sigmoid_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr7', 'in_out_ptr8'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 22, 'num_store': 9, 'num_reduction': 1, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 11600, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_abs_add_amax_bitwise_and_bitwise_not_bitwise_or_clamp_min_div_eq_frexp_gt_ldexp_le_maximum_mean_mul_ones_like_pow_reciprocal_rsub_select_sigmoid_slice_sqrt_sub_sum_unsqueeze_where_zeros_like_3(in_out_ptr0, in_out_ptr1, in_out_ptr7, in_out_ptr8, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr4, out_ptr12, out_ptr13, out_ptr14, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 100
    r0_numel = 25
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*r0_1 + 50*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr0 + (1 + 2*r0_1 + 50*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr0 + (50*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (10 + 50*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (20 + 50*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (30 + 50*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (40 + 50*x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (5*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr1 + (1 + 5*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (2 + 5*x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (3 + 5*x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr1 + (4 + 5*x0), xmask, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr2 + (5*x0), xmask, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr2 + (1 + 5*x0), xmask, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr2 + (2 + 5*x0), xmask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr2 + (3 + 5*x0), xmask, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr2 + (4 + 5*x0), xmask, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr0 + (1 + 50*x0), xmask, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr0 + (3 + 50*x0), xmask, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr0 + (5 + 50*x0), xmask, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr0 + (7 + 50*x0), xmask, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr0 + (9 + 50*x0), xmask, eviction_policy='evict_last')
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
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
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
    tl.store(out_ptr0 + (x0), tmp30, xmask)
    tl.store(out_ptr4 + (x0), tmp127, xmask)
    tl.store(out_ptr12 + (x0), tmp390, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr7 + (x0), tmp406, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp412, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp418, xmask)
    tl.store(out_ptr13 + (x0), tmp424, xmask)
    tl.store(out_ptr14 + (x0), tmp433, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr8 + (x0), tmp440, xmask)
