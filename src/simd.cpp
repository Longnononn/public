#include "simd.h"
#include <iostream>

namespace Nexus {

bool SIMDInfo::sse2 = false;
bool SIMDInfo::avx = false;
bool SIMDInfo::avx2 = false;
bool SIMDInfo::avx512 = false;
bool SIMDInfo::bmi1 = false;
bool SIMDInfo::bmi2 = false;
bool SIMDInfo::popcnt = false;
bool SIMDInfo::initialized = false;

#if defined(__x86_64__) || defined(_M_X64)
    #include <cpuid.h>
    
    static void cpuid(int info[4], int function_id) {
        __cpuid_count(function_id, 0, info[0], info[1], info[2], info[3]);
    }
#endif

void SIMDInfo::detect() {
    if (initialized) return;
    
#if defined(__x86_64__) || defined(_M_X64)
    int info[4];
    
    cpuid(info, 0);
    int nIds = info[0];
    
    if (nIds >= 1) {
        cpuid(info, 1);
        sse2 = (info[3] & (1 << 26)) != 0;
        popcnt = (info[2] & (1 << 23)) != 0;
    }
    
    if (nIds >= 7) {
        cpuid(info, 7);
        avx = (info[1] & (1 << 28)) != 0;
        avx2 = (info[1] & (1 << 5)) != 0;
        bmi1 = (info[1] & (1 << 3)) != 0;
        bmi2 = (info[1] & (1 << 8)) != 0;
        avx512 = (info[1] & (1 << 16)) != 0;  // AVX-512F
    }
#endif
    
    initialized = true;
}

void SIMDInfo::print_info() {
    detect();
    std::cout << "CPU Features:" << std::endl;
    std::cout << "  SSE2:    " << (sse2 ? "YES" : "NO") << std::endl;
    std::cout << "  POPCNT:  " << (popcnt ? "YES" : "NO") << std::endl;
    std::cout << "  AVX:     " << (avx ? "YES" : "NO") << std::endl;
    std::cout << "  AVX2:    " << (avx2 ? "YES" : "NO") << std::endl;
    std::cout << "  BMI1:    " << (bmi1 ? "YES" : "NO") << std::endl;
    std::cout << "  BMI2:    " << (bmi2 ? "YES" : "NO") << std::endl;
    std::cout << "  AVX-512: " << (avx512 ? "YES" : "NO") << std::endl;
}

bool SIMDInfo::hasSSE2() { detect(); return sse2; }
bool SIMDInfo::hasAVX() { detect(); return avx; }
bool SIMDInfo::hasAVX2() { detect(); return avx2; }
bool SIMDInfo::hasAVX512() { detect(); return avx512; }
bool SIMDInfo::hasBMI1() { detect(); return bmi1; }
bool SIMDInfo::hasBMI2() { detect(); return bmi2; }
bool SIMDInfo::hasPOPCNT() { detect(); return popcnt; }

#ifdef USE_AVX2
void AVX2Ops::relu(int16_t* output, const int16_t* input, int size) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max_val = _mm256_set1_epi16(127);  // For INT8 quantization
    
    for (int i = 0; i < size; i += 16) {
        __m256i v = _mm256_load_si256((__m256i*)&input[i]);
        v = _mm256_max_epi16(v, zero);
        v = _mm256_min_epi16(v, max_val);
        _mm256_store_si256((__m256i*)&output[i], v);
    }
}

void AVX2Ops::add_vectors(int16_t* RESTRICT a, const int16_t* RESTRICT b, int size) {
    for (int i = 0; i < size; i += 16) {
        __m256i va = _mm256_load_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_load_si256((__m256i*)&b[i]);
        va = _mm256_add_epi16(va, vb);
        _mm256_store_si256((__m256i*)&a[i], va);
    }
}

void AVX2Ops::subtract_vectors(int16_t* RESTRICT a, const int16_t* RESTRICT b, int size) {
    for (int i = 0; i < size; i += 16) {
        __m256i va = _mm256_load_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_load_si256((__m256i*)&b[i]);
        va = _mm256_sub_epi16(va, vb);
        _mm256_store_si256((__m256i*)&a[i], va);
    }
}
#endif

} // namespace Nexus
