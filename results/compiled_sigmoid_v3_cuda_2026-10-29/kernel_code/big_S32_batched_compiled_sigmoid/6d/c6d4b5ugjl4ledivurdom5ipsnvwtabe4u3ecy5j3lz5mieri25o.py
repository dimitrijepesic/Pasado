
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'out_ptr1': '*fp64', 'out_ptr3': '*fp64', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_diagonal_max_maximum_min_nan_to_num_neg_slice_view_12', 'mutated_arg_names': ['out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C589BBBF6C9873727B0988102333D2E955AFACC9DF0BF04471F83FE48A831D9A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_diagonal_max_maximum_min_nan_to_num_neg_slice_view_12(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 1024)
    x2 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (4*x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3*x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (1 + 3*x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (2 + 3*x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr2 + (3*x0), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr2 + (1 + 3*x0), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr2 + (2 + 3*x0), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = triton_helpers.minimum(tmp0, tmp1)
    tmp8 = triton_helpers.minimum(tmp7, tmp3)
    tmp9 = triton_helpers.minimum(tmp8, tmp5)
    tmp10 = -tmp9
    tmp11 = triton_helpers.maximum(tmp6, tmp10)
    tmp13 = tl.full([1], float("inf"), tl.float64)
    tmp14 = tmp12 == tmp13
    tmp15 = tl.full([1], float("-inf"), tl.float64)
    tmp16 = tmp12 == tmp15
    tmp17 = libdevice.isnan(tmp12).to(tl.int1)
    tmp18 = tl.where(tmp17, tmp15, tmp12)
    tmp19 = tl.full([1], -1.7976931348623157e+308, tl.float64)
    tmp20 = tl.where(tmp16, tmp19, tmp18)
    tmp21 = tl.full([1], 1.7976931348623157e+308, tl.float64)
    tmp22 = tl.where(tmp14, tmp21, tmp20)
    tmp24 = tmp23 == tmp13
    tmp25 = tmp23 == tmp15
    tmp26 = libdevice.isnan(tmp23).to(tl.int1)
    tmp27 = tl.where(tmp26, tmp15, tmp23)
    tmp28 = tl.where(tmp25, tmp19, tmp27)
    tmp29 = tl.where(tmp24, tmp21, tmp28)
    tmp30 = triton_helpers.maximum(tmp22, tmp29)
    tmp32 = tmp31 == tmp13
    tmp33 = tmp31 == tmp15
    tmp34 = libdevice.isnan(tmp31).to(tl.int1)
    tmp35 = tl.where(tmp34, tmp15, tmp31)
    tmp36 = tl.where(tmp33, tmp19, tmp35)
    tmp37 = tl.where(tmp32, tmp21, tmp36)
    tmp38 = triton_helpers.maximum(tmp30, tmp37)
    tmp39 = triton_helpers.maximum(tmp11, tmp38)
    tmp41 = tmp40 == tmp13
    tmp42 = tmp40 == tmp15
    tmp43 = libdevice.isnan(tmp40).to(tl.int1)
    tmp44 = tl.where(tmp43, tmp15, tmp40)
    tmp45 = tl.where(tmp42, tmp19, tmp44)
    tmp46 = tl.where(tmp41, tmp21, tmp45)
    tmp48 = tmp47 == tmp13
    tmp49 = tmp47 == tmp15
    tmp50 = libdevice.isnan(tmp47).to(tl.int1)
    tmp51 = tl.where(tmp50, tmp15, tmp47)
    tmp52 = tl.where(tmp49, tmp19, tmp51)
    tmp53 = tl.where(tmp48, tmp21, tmp52)
    tmp54 = triton_helpers.maximum(tmp46, tmp53)
    tmp56 = tmp55 == tmp13
    tmp57 = tmp55 == tmp15
    tmp58 = libdevice.isnan(tmp55).to(tl.int1)
    tmp59 = tl.where(tmp58, tmp15, tmp55)
    tmp60 = tl.where(tmp57, tmp19, tmp59)
    tmp61 = tl.where(tmp56, tmp21, tmp60)
    tmp62 = triton_helpers.maximum(tmp54, tmp61)
    tmp63 = triton_helpers.maximum(tmp39, tmp62)
    tl.store(out_ptr1 + (x0), tmp63, None)
    tl.store(out_ptr3 + (1048576 + 1024*ks0 + 1025*x1 + 2097152*x2 + 1024*ks0*x2), tmp63, None)
