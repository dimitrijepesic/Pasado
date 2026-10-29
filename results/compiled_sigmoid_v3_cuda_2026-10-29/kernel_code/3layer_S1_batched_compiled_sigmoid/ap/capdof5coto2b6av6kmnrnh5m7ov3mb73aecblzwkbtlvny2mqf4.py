
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'in_ptr3': '*fp64', 'out_ptr0': '*fp64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_clone_copy_div_expand_mul_repeat_rsub_slice_stack_sub_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 3200, 'x': 80000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_clone_copy_div_expand_mul_repeat_rsub_slice_stack_sub_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100
    xnumel = 50
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 2)
    x2 = xindex // 2
    y0 = yindex
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(x2 // 5, [YBLOCK, XBLOCK])
    tmp6 = tl.full([1, 1], 2, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([1, 1], 4.0, tl.float64)
    tmp11 = tl.broadcast_to(x2 // 5, [YBLOCK, XBLOCK])
    tmp12 = tmp11.to(tl.float64)
    tmp13 = tmp10 - tmp12
    tmp14 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 - tmp14
    tmp16 = tl.full([1, 1], 0.25, tl.float64)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp13 * tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp22 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp23 - tmp22
    tmp25 = tl.full([1, 1], 0.25, tl.float64)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp5.to(tl.float64)
    tmp28 = tmp27 * tmp26
    tmp29 = tmp22 + tmp28
    tmp30 = tl.where(tmp7, tmp21, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp4, tmp30, tmp31)
    tmp33 = tmp0 >= tmp3
    tmp34 = tl.full([1, 1], 2, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tl.broadcast_to(((x3 // 2) % 5), [YBLOCK, XBLOCK])
    tmp37 = tl.full([1, 1], 2, tl.int64)
    tmp38 = tmp36 >= tmp37
    tmp39 = tmp38 & tmp33
    tmp40 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp39 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full([1, 1], 4.0, tl.float64)
    tmp42 = tl.broadcast_to(((x3 // 2) % 5), [YBLOCK, XBLOCK])
    tmp43 = tmp42.to(tl.float64)
    tmp44 = tmp41 - tmp43
    tmp45 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp39 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp40 - tmp45
    tmp47 = tl.full([1, 1], 0.25, tl.float64)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp44 * tmp48
    tmp50 = tmp40 - tmp49
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp39, tmp50, tmp51)
    tmp53 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK])), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp54 - tmp53
    tmp56 = tl.full([1, 1], 0.25, tl.float64)
    tmp57 = tmp55 * tmp56
    tmp58 = tmp36.to(tl.float64)
    tmp59 = tmp58 * tmp57
    tmp60 = tmp53 + tmp59
    tmp61 = tl.where(tmp38, tmp52, tmp60)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp33, tmp61, tmp62)
    tmp64 = tl.where(tmp4, tmp32, tmp63)
    tl.store(out_ptr0 + (x3 + 50*y0), tmp64, xmask & ymask)
