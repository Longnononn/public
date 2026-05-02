#pragma once

#include "types.h"
#include <cstdint>

// SIMD detection and utilities for chess engine
// Supports SSE, AVX2, and AVX-512

namespace Nexus {

// Feature detection
struct SIMDInfo {
    static bool hasSSE2();
    static bool hasAVX();
    static bool hasAVX2();
    static bool hasAVX512();
    static bool hasBMI1();
    static bool hasBMI2();
    static bool hasPOPCNT();
    
    static void detect();
    static void print_info();
    
private:
    static bool sse2, avx, avx2, avx512, bmi1, bmi2, popcnt;
    static bool initialized;
};

// Fast population count using hardware instruction
inline int popcount_hardware(u64 b) {
    #if defined(_MSC_VER)
        return (int)__popcnt64(b);
    #else
        return __builtin_popcountll(b);
    #endif
}

// BMI2 PEXT/PDEP for magic bitboard index calculation
#ifdef USE_BMI2
    #if defined(_MSC_VER)
        #include <intrin.h>
        #define pext_u64 _pext_u64
        #define pdep_u64 _pdep_u64
    #else
        #define pext_u64 __builtin_ia32_pext_di
        #define pdep_u64 __builtin_ia32_pdep_di
    #endif
#endif

// Prefetch hints
#ifdef __GNUC__
    #define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
    #define PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define PREFETCH_WRITE(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
    #define PREFETCH(addr) ((void)0)
    #define PREFETCH_WRITE(addr) ((void)0)
#endif

// Branch prediction hints
#ifdef __GNUC__
    #define LIKELY(x) __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
#endif

// Force inline for critical paths
#if defined(__GNUC__)
    #define FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline
#endif

// No alias/ restrict pointer for better optimization
#ifdef __GNUC__
    #define RESTRICT __restrict
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#else
    #define RESTRICT
#endif

// Cache line size
constexpr int CACHE_LINE_SIZE = 64;

// Align to cache line
#define CACHE_ALIGN alignas(CACHE_LINE_SIZE)

// AVX2-optimized operations for NNUE evaluation
#ifdef USE_AVX2
    #include <immintrin.h>
    
    // AVX2 256-bit operations for neural network inference
    struct AVX2Ops {
        static void relu(int16_t* output, const int16_t* input, int size);
        static void matvec_accumulate(int32_t* acc, const int8_t* weights, 
                                       const int16_t* input, int inSize, int outSize);
        static void add_vectors(int16_t* RESTRICT a, const int16_t* RESTRICT b, int size);
        static void subtract_vectors(int16_t* RESTRICT a, const int16_t* RESTRICT b, int size);
    };
#endif

// SSE4.1 operations (fallback)
#ifdef USE_SSE41
    #include <nmmintrin.h>
    
    struct SSE41Ops {
        static int dot_product(const int16_t* a, const int16_t* b, int size);
    };
#endif

} // namespace Nexus
