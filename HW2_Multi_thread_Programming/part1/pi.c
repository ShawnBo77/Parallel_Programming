#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> // 使用固定寬度的整數類型，如 uint64_t
#include <time.h>
#include <immintrin.h> // 引入 Intel SIMD Intrinsics，特別是 AVX/AVX2
#include <unistd.h>
#include <string.h>

typedef struct {
    long long tosses_per_thread;
    uint64_t seed;
    long long local_hits;
    int thread_id;
} thread_data_t;

// --- PRNG state and functions ---
// __m256i 是 256-bit integer vector(可以存 8 個 int)
typedef struct {
    __m256i s0;
    __m256i s1;
} xorshiftr128plus_state;


// 生成 8 個不同的 64 位元初始狀態值當 s0 和 s1 向量。
void avx2_xorshiftr128plus_init(xorshiftr128plus_state* state, uint64_t seed) {
    uint64_t s[4];
    s[0] = seed ^ 0xabcdabcdabcdabcd;
    s[1] = seed * 0x1234567890abcdef;
    s[2] = seed + 0xfedcba9876543210;
    s[3] = ~seed;
    state->s0 = _mm256_loadu_si256((__m256i*)s);
    s[0] = seed ^ 0x1111111111111111;
    s[1] = seed * 0x2222222222222222;
    s[2] = seed + 0x3333333333333333;
    s[3] = ~seed + 0x4444444444444444;
    state->s1 = _mm256_loadu_si256((__m256i*)s);
}

// 256-bit 可以裝 8 個 32-bit 整數
// static inline 建議 compiler 將函式展開，提高效能。
// 實作 xorshiftr128+ 生成 8 個隨機的 32-bit 整數
static inline __m256i avx2_xorshiftr128plus(xorshiftr128plus_state* state) {

    __m256i x = state->s0;
    const __m256i y = state->s1;
    state->s0 = y;
    
    x = _mm256_xor_si256(x, _mm256_slli_epi64(x, 23)); // x ^= x << 23
    x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 17)); // x ^= x >> 17
    x = _mm256_xor_si256(x, y);                        // x ^= y
    
    state->s1 = _mm256_add_epi64(x, y);
    return x;
}

void* worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;

    // --- CPU affinity ---
    // 有助於減少 cache miss，但也可能造成負載不均衡
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset); // 清空 CPU 集合
    // CPU_SET(data->thread_id, &cpuset); // 將指定 CPU core 加入集合
    // pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset); // 綁定當前 thread 到該 CPU
    // int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    // printf("num_cores: %d", num_cores);
    // int core_id = data->thread_id % num_cores; // 確保在範圍內

    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset); // 清空 CPU 集合
    // CPU_SET(core_id, &cpuset); // 將指定 CPU core 加入集合

    // int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset); // 綁定當前 thread 到該 CPU
    // if (ret != 0) {
    //     fprintf(stderr, "Warning: Failed to set affinity for thread %d (core %d): %s\n",
    //             data->thread_id, core_id, strerror(ret));
    // }

    long long hits = 0;
    xorshiftr128plus_state rng_state;
    avx2_xorshiftr128plus_init(&rng_state, data->seed);

    // __m256 是裝 8 個 32-bit floating point 的 vector
    // _mm256_set1_ps 是把 256-bit 中 每 32-bit 設成固定的值
    const __m256 ones_ps = _mm256_set1_ps(1.0f); // 即 1.0f | 1.0f | 1.0f | 1.0f | 1.0f | 1.0f | 1.0f | 1.0f
    const __m256 three_ps = _mm256_set1_ps(3.0f);
    const __m256i mask_significand = _mm256_set1_epi32(0x007FFFFF); // Significand (最後 23 bits) 的 mask，從 32-bit 整數中提取最後 23 bits 作為 Significand。
    const __m256i mask_exp = _mm256_set1_epi32(0x3F800000); // Exponent 的 mask，設置浮點數的指數部分為 0 (bias=127；0+127=127；127 -> 01111111 換成 32-bit 0|011 1111 1|000...)，使其落在 [1.0, 2.0) 區間。
    
    // 採用 Loop unrolling：每次迴圈處理 32 個點 (4 組 8-float 的 AVX 向量)。
    long long loop_count = data->tosses_per_thread / 32;
    // 一次處理 32 個點 (32 個 x, 32 個 y)，即 64 個數。
    for (long long i = 0; i < loop_count; ++i) {
        // --- 生成隨機數 ---
        // 每次 avx2_xorshiftr128plus 產生 256-bit 即 8 個 32-bit 浮點數，所以總共要 call 8 次。
        __m256i r1 = avx2_xorshiftr128plus(&rng_state);
        __m256i r2 = avx2_xorshiftr128plus(&rng_state);
        __m256i r3 = avx2_xorshiftr128plus(&rng_state);
        __m256i r4 = avx2_xorshiftr128plus(&rng_state);
        __m256i r5 = avx2_xorshiftr128plus(&rng_state);
        __m256i r6 = avx2_xorshiftr128plus(&rng_state);
        __m256i r7 = avx2_xorshiftr128plus(&rng_state);
        __m256i r8 = avx2_xorshiftr128plus(&rng_state);

        // --- 轉為 [1.0, 2.0) 的浮點數 ---
        // 以 IEEE 754 標準的浮點數的二進位表示（ 1-bit Sign | 8-bit Exponent | 23-bit Significand ）。
        // _mm256_castsi256_ps 將整數轉為浮點數。
        __m256 x1_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r1, mask_significand), mask_exp));
        __m256 y1_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r2, mask_significand), mask_exp));
        __m256 x2_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r3, mask_significand), mask_exp));
        __m256 y2_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r4, mask_significand), mask_exp));
        __m256 x3_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r5, mask_significand), mask_exp));
        __m256 y3_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r6, mask_significand), mask_exp));
        __m256 x4_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r7, mask_significand), mask_exp));
        __m256 y4_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r8, mask_significand), mask_exp));

        // --- 映射到 [-1.0, 1.0) ---
        __m256 x1 = _mm256_sub_ps(_mm256_add_ps(x1_12, x1_12), three_ps);
        __m256 y1 = _mm256_sub_ps(_mm256_add_ps(y1_12, y1_12), three_ps);
        __m256 x2 = _mm256_sub_ps(_mm256_add_ps(x2_12, x2_12), three_ps);
        __m256 y2 = _mm256_sub_ps(_mm256_add_ps(y2_12, y2_12), three_ps);
        __m256 x3 = _mm256_sub_ps(_mm256_add_ps(x3_12, x3_12), three_ps);
        __m256 y3 = _mm256_sub_ps(_mm256_add_ps(y3_12, y3_12), three_ps);
        __m256 x4 = _mm256_sub_ps(_mm256_add_ps(x4_12, x4_12), three_ps);
        __m256 y4 = _mm256_sub_ps(_mm256_add_ps(y4_12, y4_12), three_ps);

        // --- 計算距離平方 ---
        __m256 d_sq1 = _mm256_add_ps(_mm256_mul_ps(x1, x1), _mm256_mul_ps(y1, y1));
        __m256 d_sq2 = _mm256_add_ps(_mm256_mul_ps(x2, x2), _mm256_mul_ps(y2, y2));
        __m256 d_sq3 = _mm256_add_ps(_mm256_mul_ps(x3, x3), _mm256_mul_ps(y3, y3));
        __m256 d_sq4 = _mm256_add_ps(_mm256_mul_ps(x4, x4), _mm256_mul_ps(y4, y4));

        // --- 比較是否 < 1 ---
        // 比較 32-bit floating point，成立回傳 32-bit 的 1，不成立則全 0
        __m256 mask1 = _mm256_cmp_ps(d_sq1, ones_ps, _CMP_LE_OQ);
        __m256 mask2 = _mm256_cmp_ps(d_sq2, ones_ps, _CMP_LE_OQ);
        __m256 mask3 = _mm256_cmp_ps(d_sq3, ones_ps, _CMP_LE_OQ);
        __m256 mask4 = _mm256_cmp_ps(d_sq4, ones_ps, _CMP_LE_OQ);

        // --- 計算命中次數 ---
        // _mm256_movemask_ps 提取 8 個 32-bit floating point 的 the most significant bit，組成 8-bit integer。
        int m1 = _mm256_movemask_ps(mask1); // 假如提取完為 [1, 0, 1, 1, 0, 0, 1, 0]， m1 = 0b10110010 (二進位)，即十進位的 178。
        int m2 = _mm256_movemask_ps(mask2);
        int m3 = _mm256_movemask_ps(mask3);
        int m4 = _mm256_movemask_ps(mask4);
        
        // __builtin_popcount() is used to count the number of set bits in long long data types
        // 計算一個整數的二進位表示中有多少個 bit 被設為 1
        hits += __builtin_popcount(m1) + __builtin_popcount(m2) + __builtin_popcount(m3) + __builtin_popcount(m4);
    }

    // --- 處理剩餘部分 ---
    long long remaining_tosses = data->tosses_per_thread % 32;
    unsigned int seed32 = (unsigned int)data->seed;
    for (long long i = 0; i < remaining_tosses; ++i) {
        float x_f = (float)rand_r(&seed32) / RAND_MAX * 2.0f - 1.0f;
        float y_f = (float)rand_r(&seed32) / RAND_MAX * 2.0f - 1.0f;
        if (x_f * x_f + y_f * y_f <= 1.0f) {
            hits++;
        }
    }

    if (remaining_tosses > 0) {
        // Mask Look up table
        static const int mask_lut[9] = {
            0b00000000,
            0b00000001,
            0b00000011,
            0b00000111,
            0b00001111,
            0b00011111,
            0b00111111,
            0b01111111,
            0b11111111
        };

        // 使用 while 迴圈處理剩餘的點，每次處理最多 8 個
        while (remaining_tosses > 0) {
            long long chunk_size = (remaining_tosses >= 8) ? 8 : remaining_tosses;
            
            // 從 LUT 中獲取對應長度的 movemask 結果遮罩
            const int len_mask = mask_lut[chunk_size];

            __m256i r1 = avx2_xorshiftr128plus(&rng_state);
            __m256i r2 = avx2_xorshiftr128plus(&rng_state);
            
            __m256 x_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r1, mask_significand), mask_exp));
            __m256 y_12 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(r2, mask_significand), mask_exp));
            
            __m256 x = _mm256_sub_ps(_mm256_add_ps(x_12, x_12), three_ps);
            __m256 y = _mm256_sub_ps(_mm256_add_ps(y_12, y_12), three_ps);

            __m256 d_sq = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
            
            __m256 cmp_mask_ps = _mm256_cmp_ps(d_sq, ones_ps, _CMP_LE_OQ);

            int cmp_mask = _mm256_movemask_ps(cmp_mask_ps);

            // 保留 chunk_size 內的值
            int final_mask = cmp_mask & len_mask;
            
            hits += __builtin_popcount(final_mask);
            
            remaining_tosses -= chunk_size;
        }
    }

    data->local_hits = hits;
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <number_of_threads> <number_of_tosses>\n", argv[0]);
        return 1;
    }
    int num_threads = atoi(argv[1]);
    long long total_tosses = atoll(argv[2]);
    if (num_threads <= 0 || total_tosses <= 0) {
        fprintf(stderr, "Number of threads and tosses must be positive.\n");
        return 1;
    }
    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];
    long long tosses_per_thread = total_tosses / num_threads;
    long long remainder = total_tosses % num_threads;
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].tosses_per_thread = tosses_per_thread + (i < remainder ? 1 : 0);
        // 0x9e3779b97f4a7c15 是 64-bit 黃金比例常數 (2^64 × (φ – 1) 的近似值) 能均勻打散數值
        thread_data[i].seed = (uint64_t)time(NULL) ^ (((uint64_t)i + 1) * 0x9e3779b97f4a7c15);
        thread_data[i].local_hits = 0;
        thread_data[i].thread_id = i;
        int rc = pthread_create(&threads[i], NULL, worker, (void*)&thread_data[i]);
        if (rc) {
            fprintf(stderr, "ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    long long total_hits = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
        total_hits += thread_data[i].local_hits;
    }

    double pi_estimate = 4.0 * total_hits / total_tosses;
    printf("%lf\n", pi_estimate);
    return 0;
}
