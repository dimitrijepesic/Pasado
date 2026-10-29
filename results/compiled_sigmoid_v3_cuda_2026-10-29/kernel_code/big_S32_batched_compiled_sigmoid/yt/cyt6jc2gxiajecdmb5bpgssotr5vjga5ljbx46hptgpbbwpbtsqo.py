
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 2},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'out_ptr1': '*fp64', 'out_ptr2': '*fp64', 'out_ptr3': '*fp64', 'out_ptr4': '*fp64', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_add_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 4, 'num_reduction': 1, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2883584, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_abs_add_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_sum_1(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float64)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 1024*ks0*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl_math.abs(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp6 = tmp5 - tmp3
    tmp7 = tmp5 + tmp3
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = -tmp6
    tmp10 = libdevice.exp(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = (tmp8 / tmp11)
    tmp13 = tl.full([1, 1], 1.0, tl.float64)
    tmp14 = tmp13 - tmp12
    tmp15 = tmp12 * tmp14
    tmp16 = -tmp7
    tmp17 = libdevice.exp(tmp16)
    tmp18 = tmp8 + tmp17
    tmp19 = (tmp8 / tmp18)
    tmp20 = tmp13 - tmp19
    tmp21 = tmp19 * tmp20
    tmp22 = triton_helpers.minimum(tmp15, tmp21)
    tmp23 = tmp22 * tmp5
    tmp24 = tmp19 + tmp12
    tmp25 = tmp7 + tmp6
    tmp26 = tmp22 * tmp25
    tmp27 = tmp24 - tmp26
    tmp28 = tl.full([1, 1], 0.5, tl.float64)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp23 + tmp29
    tmp31 = tmp19 - tmp12
    tmp32 = tmp7 - tmp6
    tmp33 = tmp22 * tmp32
    tmp34 = tmp31 - tmp33
    tmp35 = tmp34 * tmp28
    tl.store(out_ptr1 + (x3), tmp6, None)
    tl.store(out_ptr2 + (x3), tmp7, None)
    tl.store(out_ptr3 + (x3), tmp30, None)
    tl.store(out_ptr4 + (x3), tmp35, None)
