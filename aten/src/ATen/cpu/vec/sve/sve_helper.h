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

// Add an additional macro to handle the token pasting
#define CONCAT(a, b, c) a##b##c
#define IMPLEMENT_EXP_SVE(bit) \
    static inline CONCAT(svfloat, bit, _t) svexp_f##bit##_x(svbool_t pg, CONCAT(svfloat, bit, _t) x) { \
        constexpr CONCAT(float, bit, _t) cst_exp_hi = 88.3762626647950; \
        constexpr CONCAT(float, bit, _t) cst_exp_lo = -88.3762626647949; \
        constexpr CONCAT(float, bit, _t) cst_cephes_LOG2EF = 1.44269504088896341; \
        constexpr CONCAT(float, bit, _t) cst_nln2 = -0.6931471805599453; \
        const CONCAT(svfloat, bit, _t) cst_exp_p0 = svdup_n_f##bit(1.9875691500e-4); \
        const CONCAT(svfloat, bit, _t) cst_exp_p1 = svdup_n_f##bit(1.3981999507e-3); \
        const CONCAT(svfloat, bit, _t) cst_exp_p2 = svdup_n_f##bit(8.3334519073e-3); \
        const CONCAT(svfloat, bit, _t) cst_exp_p3 = svdup_n_f##bit(4.1665795894e-2); \
        const CONCAT(svfloat, bit, _t) cst_exp_p4 = svdup_n_f##bit(1.6666665459e-1); \
        const CONCAT(svfloat, bit, _t) cst_exp_p5 = svdup_n_f##bit(5.0000001201e-1); \
        const CONCAT(svfloat, bit, _t) c = svminnm_n_f##bit##_x( \
            pg, svmaxnm_n_f##bit##_x(pg, x, cst_exp_lo), cst_exp_hi); \
        const CONCAT(svfloat, bit, _t) m = svrinti_f##bit##_x(pg, svmul_n_f##bit##_x(pg, c, cst_cephes_LOG2EF)); \
        const CONCAT(svfloat, bit, _t) r = svmla_n_f##bit##_x(pg, c, m, cst_nln2); \
        const CONCAT(svfloat, bit, _t) r2 = svmul_f##bit##_x(pg, r, r); \
        const CONCAT(svfloat, bit, _t) r3 = svmul_f##bit##_x(pg, r2, r); \
        CONCAT(svfloat, bit, _t) y = svmla_f##bit##_x(pg, cst_exp_p1, cst_exp_p0, r); \
        y = svmla_f##bit##_x(pg, cst_exp_p2, y, r); \
        CONCAT(svfloat, bit, _t) y1 = svmla_f##bit##_x(pg, cst_exp_p4, cst_exp_p3, r); \
        y1 = svmla_f##bit##_x(pg, cst_exp_p5, y1, r); \
        y = svmla_f##bit##_x(pg, y1, y, r3); \
        const CONCAT(svfloat, bit, _t) y2 = svadd_n_f##bit##_x(pg, r, 1.0); \
        y = svmla_f##bit##_x(pg, y2, y, r2); \
        return svmax_f##bit##_x(pg, svscale_f##bit##_x(pg, y, svcvt_s##bit##_f##bit##_x(pg, m)), x); \
    }

// Now invoke the macro for 32-bit and 16-bit versions
IMPLEMENT_EXP_SVE(32)
IMPLEMENT_EXP_SVE(16)

#endif // defined(CPU_CAPABILITY_SVE)
