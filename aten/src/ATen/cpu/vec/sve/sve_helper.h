#pragma once

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/vec_base.h>

#if defined(CPU_CAPABILITY_SVE)

// Define the data type of VLS(vector-length specific).
typedef svbool_t vls_pred_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint8_t vls_int8_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint16_t vls_int16_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint32_t vls_int32_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint64_t vls_int64_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint8_t vls_uint8_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint16_t vls_uint16_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint32_t vls_uint32_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint64_t vls_uint64_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat16_t vls_float16_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat32_t vls_float32_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat64_t vls_float64_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));

#define ptrue svptrue_b8()
#define ZERO_S8 svdup_n_s8(0)
#define ZERO_S16 svdup_n_s16(0)
#define ZERO_S32 svdup_n_s32(0)
#define ZERO_S64 svdup_n_s64(0)
#define ZERO_U8 svdup_n_u8(0)
#define ZERO_U16 svdup_n_u16(0)
#define ZERO_U32 svdup_n_u32(0)
#define ZERO_U64 svdup_n_u64(0)
#define ZERO_F16 svdup_n_f16(0.f)
#define ZERO_F32 svdup_n_f32(0.f)
#define ZERO_F64 svdup_n_f64(0.0)
#define ONE_S8  svdup_n_s8(1)
#define ONE_S16 svdup_n_s16(1)
#define ONE_S32 svdup_n_s32(1)
#define ONE_S64 svdup_n_s64(1)
#define ONE_U8 svdup_n_u8(1)
#define ONE_U16 svdup_n_u16(1)
#define ONE_U32 svdup_n_u32(1)
#define ONE_U64 svdup_n_u64(1)
#define ONE_F16 svdup_n_f16(1.f)
#define ONE_F32 svdup_n_f32(1.f)
#define ONE_F64 svdup_n_f64(1.0)
#define ALL_S8_TRUE_MASK svdup_n_s8(0xff)
#define ALL_S8_FALSE_MASK svdup_n_s8(0x0)
#define ALL_S16_TRUE_MASK svdup_n_s16(0xffff)
#define ALL_S16_FALSE_MASK svdup_n_s16(0x0)
#define ALL_S32_TRUE_MASK svdup_n_s32(0xffffffff)
#define ALL_S32_FALSE_MASK svdup_n_s32(0x0)
#define ALL_S64_TRUE_MASK svdup_n_s64(0xffffffffffffffff)
#define ALL_S64_FALSE_MASK svdup_n_s64(0x0)
#define ALL_U8_TRUE_MASK svdup_n_u8(0x01)
#define ALL_U8_FALSE_MASK svdup_n_u8(0x00)
#define ALL_F16_TRUE_MASK svreinterpret_f16_s16(ALL_S16_TRUE_MASK)
#define ALL_F16_FALSE_MASK svreinterpret_f16_s16(ALL_S16_FALSE_MASK)
#define ALL_F32_TRUE_MASK svreinterpret_f32_s32(ALL_S32_TRUE_MASK)
#define ALL_F32_FALSE_MASK svreinterpret_f32_s32(ALL_S32_FALSE_MASK)
#define ALL_F64_TRUE_MASK svreinterpret_f64_s64(ALL_S64_TRUE_MASK)
#define ALL_F64_FALSE_MASK svreinterpret_f64_s64(ALL_S64_FALSE_MASK)

inline svfloat32_t svexp_f32_x(svbool_t pg, svfloat32_t x) {
    const auto c1 = svreinterpret_f32_u32(svdup_n_u32(0x3f7ffff6)); // x^1: 0x1.ffffecp-1f
    const auto c2 = svreinterpret_f32_u32(svdup_n_u32(0x3efffedb)); // x^2: 0x1.fffdb6p-2f
    const auto c3 = svreinterpret_f32_u32(svdup_n_u32(0x3e2aaf33)); // x^3: 0x1.555e66p-3f
    const auto c4 = svreinterpret_f32_u32(svdup_n_u32(0x3d2b9f17)); // x^4: 0x1.573e2ep-5f
    const auto c5 = svreinterpret_f32_u32(svdup_n_u32(0x3c072010)); // x^5: 0x1.0e4020p-7f

    const auto shift   = svreinterpret_f32_u32(svdup_n_u32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
    const auto inv_ln2 = svreinterpret_f32_u32(svdup_n_u32(0x3fb8aa3b)); // 1 / ln(2) = 0x1.715476p+0f
    const auto neg_ln2_hi =
        svreinterpret_f32_u32(svdup_n_u32(0xbf317200)); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    const auto neg_ln2_lo =
        svreinterpret_f32_u32(svdup_n_u32(0xb5bfbe8e)); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

    const auto inf       = svdup_n_f32(std::numeric_limits<float>::infinity());
    const auto max_input = svdup_n_f32(88.37f); // Approximately ln(2^127.5)
    const auto zero      = svdup_n_f32(0.f);
    const auto min_input = svdup_n_f32(-86.64f); // Approximately ln(2^-125)

    // Range reduction:
    //   e^x = 2^n * e^r
    // where:
    //   n = floor(x / ln(2))
    //   r = x - n * ln(2)
    //
    // By adding x / ln(2) with 2^23 + 127 (shift):
    //   * As FP32 fraction part only has 23-bits, the addition of 2^23 + 127 forces decimal part
    //     of x / ln(2) out of the result. The integer part of x / ln(2) (i.e. n) + 127 will occupy
    //     the whole fraction part of z in FP32 format.
    //     Subtracting 2^23 + 127 (shift) from z will result in the integer part of x / ln(2)
    //     (i.e. n) because the decimal part has been pushed out and lost.
    //   * The addition of 127 makes the FP32 fraction part of z ready to be used as the exponent
    //     in FP32 format. Left shifting z by 23 bits will result in 2^n.
    const auto z     = svmla_f32_z(pg, shift, x, inv_ln2);
    const auto n     = svsub_f32_z(pg, z, shift);
    const auto scale = svreinterpret_f32_u32(svlsl_n_u32_z(pg, svreinterpret_u32_f32(z), 23)); // 2^n

    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy beyond FP32.
    // This outperforms longer Taylor series (3-4 tabs) both in term of accuracy and performance.
    const auto r_hi = svmla_f32_z(pg, x, n, neg_ln2_hi);
    const auto r    = svmla_f32_z(pg, r_hi, n, neg_ln2_lo);

    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    const auto r2 = svmul_f32_z(pg, r, r);

    const auto p1     = svmul_f32_z(pg, c1, r);
    const auto p23    = svmla_f32_z(pg, c2, c3, r);
    const auto p45    = svmla_f32_z(pg, c4, c5, r);
    const auto p2345  = svmla_f32_z(pg, p23, p45, r2);
    const auto p12345 = svmla_f32_z(pg, p1, p2345, r2);

    auto poly = svmla_f32_z(pg, scale, p12345, scale);

    // Handle underflow and overflow.
    poly = svsel_f32(svcmplt_f32(pg, x, min_input), zero, poly);
    poly = svsel_f32(svcmpgt_f32(pg, x, max_input), inf, poly);

    return poly;
}

#endif // defined(CPU_CAPABILITY_SVE)
