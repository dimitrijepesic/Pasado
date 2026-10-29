
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
