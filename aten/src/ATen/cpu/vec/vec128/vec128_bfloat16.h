#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#define CONVERT_NON_VECTORIZED_INIT(type, name) \
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_##name##_float(const Vectorized<type>& a) { \
  constexpr int64_t K = Vectorized<type>::size(); \
  __at_align__ float arr[K]; \
  __at_align__ type arr2[K]; \
  a.store(arr2); \
  convert(arr2, arr, K); \
  return std::make_tuple( \
      Vectorized<float>::loadu(arr), \
      Vectorized<float>::loadu(arr + Vectorized<float>::size())); \
} \
inline Vectorized<type> convert_float_##name(const Vectorized<float>& a, const Vectorized<float>& b) { \
  constexpr int64_t K = Vectorized<type>::size(); \
  __at_align__ float arr[K]; \
  __at_align__ type arr2[K]; \
  a.store(arr); \
  b.store(arr + Vectorized<float>::size()); \
  convert(arr, arr2, K); \
  return Vectorized<type>::loadu(arr2); \
}
CONVERT_NON_VECTORIZED_INIT(BFloat16, bfloat16);
#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__) && !defined(CPU_CAPABILITY_SVE128)
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_half_float(const Vectorized<Half>& a) {
  static_assert(Vectorized<Half>::size() == 2 * Vectorized<float>::size());
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  float16x8x2_t arr = a;
  float16x8_t x = arr.val[0];
  float16x8_t y = arr.val[1];
#else
  auto arr = reinterpret_cast<const float16_t*>(a.operator const Half*());
  float16x8_t x = vld1q_f16(arr);
  float16x8_t y = vld1q_f16(arr + Vectorized<float>::size());
#endif
  float32x4_t x1 = vcvt_f32_f16(vget_low_f16(x));
  float32x4_t x2 = vcvt_f32_f16(vget_high_f16(x));
  float32x4_t y1 = vcvt_f32_f16(vget_low_f16(y));
  float32x4_t y2 = vcvt_f32_f16(vget_high_f16(y));
  return { Vectorized<float>(x1, x2), Vectorized<float>(y1, y2) };
}
inline Vectorized<Half> convert_float_half(const Vectorized<float>& a, const Vectorized<float>& b) {
  static_assert(Vectorized<Half>::size() == 2 * Vectorized<float>::size());
  float32x4x2_t x = a;
  float32x4x2_t y = b;
  float16x4_t x1 = vcvt_f16_f32(x.val[0]);
  float16x4_t x2 = vcvt_f16_f32(x.val[1]);
  float16x4_t y1 = vcvt_f16_f32(y.val[0]);
  float16x4_t y2 = vcvt_f16_f32(y.val[1]);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  return Vectorized<Half>(vcombine_f16(x1, x2), vcombine_f16(y1, y2));
#else
  Vectorized<Half> rc;
  auto arr = reinterpret_cast<float16_t*>(rc.operator Half*());
  vst1q_f16(arr, vcombine_f16(x1, x2));
  vst1q_f16(arr + Vectorized<float>::size(), vcombine_f16(y1, y2));
  return rc;
#endif
}
#else
CONVERT_NON_VECTORIZED_INIT(Half, half);
#endif

#define LOAD_FP32_NON_VECTORIZED_INIT(type, name) \
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out) { \
  __at_align__ float values[Vectorized<float>::size()]; \
  for (const auto k : c10::irange(Vectorized<float>::size())) { \
    values[k] = data[k]; \
  } \
  out = Vectorized<float>::loadu(values); \
} \
\
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out1, Vectorized<float>& out2) { \
  load_fp32_from_##name(data, out1); \
  data += Vectorized<float>::size(); \
  load_fp32_from_##name(data, out2); \
}
LOAD_FP32_NON_VECTORIZED_INIT(BFloat16, bf16);
LOAD_FP32_NON_VECTORIZED_INIT(Half, fp16);

}} // namsepace at::vec::CPU_CAPABILITY

#pragma GCC diagnostic pop
