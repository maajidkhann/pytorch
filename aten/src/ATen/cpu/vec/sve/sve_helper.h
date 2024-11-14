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

  inline svfloat32_t svexp_f32_x(svbool_t pg, svfloat32_t src) {
    // Constants
        const auto log2_e = svdup_n_f32(1.0f / std::log(2.0f));
        //const auto ln2 = svdup_n_f32(0.6931473921f);
        //const auto half_ln2_sq = svdup_n_f32(0.2413862043f);

        const auto ln2 = svdup_n_f32(0.6931471805599453f);
        const auto half_ln2_sq = svdup_n_f32(0.2413862295f); // More precise value

        const auto not_mask17 = svdup_n_u32(~((1u << 17) - 1));
        const auto one = svdup_n_f32(1.0f);
        const auto zero = svdup_n_f32(0.0f);
        const auto inf = svdup_n_f32(std::numeric_limits<float>::infinity());
        const auto max_input = svdup_n_f32(88.37f);                             // Approximately ln(2^127.5)
        const auto min_input = svdup_n_f32(-86.64f);                            // Approximately ln(2^-125)

        // Algorithm starts here
        svfloat32_t t0 = svmul_f32_z(pg, src, log2_e);                          // y = x * log2(e)
        svfloat32_t t1 = svrintm_f32_z(pg, t0);                                 // rount to int (float)
        svint32_t t2 = svcvt_s32_f32_z(pg, t1);                                 // n

        t1 = svsub_f32_z(pg, t0, t1);                                           // a = y - floor(y)
        t1 = svadd_f32_z(pg, t1, one);                                          // b = a + 1

        svuint32_t t3 = svlsr_n_u32_z(pg, svreinterpret_u32_f32(t1), 17);       // v = b >> 17 (u32)
        svfloat32_t t4 = svexpa_f32(t3);                                        // c = fexpa(v)
        t4 = svscale_f32_z(pg, t4, t2);                                         // fexpa(v) * 2^(n)

        // and_(t2.d, t1.d, not_mask17.d)
        svfloat32_t t5 = svreinterpret_f32_u32(svand_u32_z(pg, svreinterpret_u32_f32(t1), not_mask17));
        t5 = svsub_f32_z(pg, t1, t5);                                           // z
        t0 = svmla_f32_z(pg, ln2, t5, half_ln2_sq);                             // ln2 + half_ln2_sq * z
        t0 = svmla_f32_z(pg, one, t5, t0);                                      // 1 + (ln2 * z) + (half_ln2_sq * z * z)
        t0 = svmul_f32_z(pg, t0, t4);                                           // Final result

        return t0;
    }

#endif // defined(CPU_CAPABILITY_SVE)
