#pragma once
#include <string>
#include <tuple>
#include <iostream>

#include "types.h"

#include <hwy/highway.h>


HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
static inline void simd_xor(uint32_t* dst, const uint32_t* src, size_t count) {
  using namespace hwy;
  const ScalableTag<uint32_t> d;
  for (size_t i = 0; i < count; i += Lanes(d)) {
    auto v_dst = LoadU(d, dst + i);
    auto v_src = LoadU(d, src + i);
    auto v_xor = Xor(v_dst, v_src);
    StoreU(v_xor, d, dst + i);
  }
}
static inline void simd_xor_aligned(uint32_t* dst, const uint32_t* src, size_t count) {
  using namespace hwy;
  const ScalableTag<uint32_t> d;
  for (size_t i = 0; i < count; i += Lanes(d)) {
    auto v_dst = Load(d, dst + i);
    auto v_src = Load(d, src + i);
    auto v_xor = Xor(v_dst, v_src);
    Store(v_xor, d, dst + i);
  }
}
} // namespace HWY_NAMESPACE
} // namespace hwy
HWY_AFTER_NAMESPACE();


/**
 * Cast a double to unsigned long long with epsilon adjustment.
 * @param d         the double to cast.
 * @param epsilon   optional parameter representing the epsilon to use.
 */
unsigned long long int double_to_ull(double d, double epsilon = 0.00000001);

/**
 * A function N x N -> N that implements a non-self-edge pairing function
 * that does not respect order of inputs.
 * That is, (2,2) would not be a valid input. (1,3) and (3,1) would be treated as
 * identical inputs.
 * @return i + j*(j-1)/2
 */
edge_id_t nondirectional_non_self_edge_pairing_fn(node_id_t i, node_id_t j);

/**
 * Inverts the nondirectional non-SE pairing function.
 * @param idx
 * @return the pair, with left and right ordered lexicographically.
 */
Edge inv_nondir_non_self_edge_pairing_fn(edge_id_t idx);

/**
 * Concatenates two node ids to form an edge ids
 * @return (i << 32) & j
 */
edge_id_t concat_pairing_fn(node_id_t i, node_id_t j);

/**
 * Inverts the concat pairing function.
 * @param idx
 * @return the pair, with left and right ordered lexicographically.
 */
Edge inv_concat_pairing_fn(edge_id_t idx);

#define likely_if(x) if(__builtin_expect((bool)(x), true))
#define unlikely_if(x) if (__builtin_expect((bool)(x), false))

inline static void set_bit(vec_t &t, int position) {
  t |= 1 << position;
}

inline static void clear_bit(vec_t &t, int position) {
  t &= ~(1 << position);
}