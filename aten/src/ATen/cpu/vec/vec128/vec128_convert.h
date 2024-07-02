#pragma once

#include <ATen/cpu/vec/functional_bfloat16.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_convert.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

template <typename src_t>
struct VecConvert<
    float,
    1,
    src_t,
    1,
    typename std::enable_if_t<is_reduced_floating_point_v<src_t>, void>> {
  static inline VectorizedN<float, 1> apply(const VectorizedN<src_t, 1>& src) {
    auto [res_vec1, res_vec2] = convert_to_float<src_t>(src[0]);
    return res_vec1;
  }
};

template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    float,
    1,
    typename std::enable_if_t<is_reduced_floating_point_v<dst_t>, void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 1>& src) {
    return convert_from_float<dst_t>(src[0], src[0]);
  }
};

} // namespace CPU_CAPABILITY
} // namespace at::vec
