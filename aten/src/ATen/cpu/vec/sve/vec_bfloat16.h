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
#if defined(__aarch64__) && defined(CPU_CAPABILITY_SVE) && !defined(C10_MOBILE) && !defined(__CUDACC__)
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_half_float(const Vectorized<Half>& a) {
  static_assert(Vectorized<Half>::size() == 2 * Vectorized<float>::size());
  svbool_t pg = svptrue_b16();

  auto arr = reinterpret_cast<const __fp16*>(a.operator const Half*());
  svfloat16_t x = svld1(pg, arr);
  svfloat16_t y = svld1(pg, arr + svcntw());

  svfloat32_t x1 = svcvt_f32_f16_x(pg, x);
  svfloat32_t y1 = svcvt_f32_f16_x(pg, y);

  return { Vectorized<float>(x1), Vectorized<float>(y1) };
}

inline Vectorized<Half> convert_float_half(const Vectorized<float>& a, const Vectorized<float>& b) {
  static_assert(Vectorized<Half>::size() == 2 * Vectorized<float>::size());
  svbool_t pg = svptrue_b16();

  alignas(64) float a_data[Vectorized<float>::size()];
  alignas(64) float b_data[Vectorized<float>::size()];
  a.store(a_data);
  b.store(b_data);

  svfloat32_t x1 = svld1(pg, a_data);
  svfloat32_t x2 = svld1(pg, a_data + svcntw());
  svfloat32_t y1 = svld1(pg, b_data);
  svfloat32_t y2 = svld1(pg, b_data + svcntw());

  svfloat16_t x1_half = svcvt_f16_f32_x(pg, x1);
  svfloat16_t x2_half = svcvt_f16_f32_x(pg, x2);
  svfloat16_t y1_half = svcvt_f16_f32_x(pg, y1);
  svfloat16_t y2_half = svcvt_f16_f32_x(pg, y2);

  Vectorized<Half> rc;
  auto arr = reinterpret_cast<__fp16*>(rc.operator Half*());
  svst1(pg, arr, x1_half);
  svst1(pg, arr + svcntw(), x2_half);
  svst1(pg, arr + 2*svcntw(), y1_half);
  svst1(pg, arr + 3*svcntw(), y2_half);
  return rc;
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