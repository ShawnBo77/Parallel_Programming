#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2
#include <smmintrin.h> // SSE4.1

uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// The core computation function using SSE2 with single-precision floats.
long long int compute_pi_hits_sse(long long int tosses_per_process, int world_rank) {
    long long int hits = 0;

    // Initialize PRNG state with a unique seed for each MPI rank.
    uint32_t rng_state = (uint32_t)time(NULL) ^ (uint32_t)world_rank;

    // Constants for fast uint32 to float [0, 1) conversion
    const __m128i mask_23 = _mm_set1_epi32(0x007FFFFF); // Mask for mantissa
    const __m128i exp_127 = _mm_set1_epi32(0x3F800000); // Exponent for 1.0
    const __m128 ones_ps = _mm_set1_ps(1.0f);

    long long loop_count = tosses_per_process / 16;
    for (long long i = 0; i < loop_count; ++i) {
        // --- Generate 32 random uint32 for 16 points ---
        uint32_t r[32];
        for(int k=0; k<32; ++k) r[k] = xorshift32(&rng_state);

        // --- Load random integers into SSE registers ---
        __m128i r_x1 = _mm_loadu_si128((__m128i*)&r[0]);
        __m128i r_y1 = _mm_loadu_si128((__m128i*)&r[4]);
        __m128i r_x2 = _mm_loadu_si128((__m128i*)&r[8]);
        __m128i r_y2 = _mm_loadu_si128((__m128i*)&r[12]);
        __m128i r_x3 = _mm_loadu_si128((__m128i*)&r[16]);
        __m128i r_y3 = _mm_loadu_si128((__m128i*)&r[20]);
        __m128i r_x4 = _mm_loadu_si128((__m128i*)&r[24]);
        __m128i r_y4 = _mm_loadu_si128((__m128i*)&r[28]);
        
        // --- Convert uint32 to float in [0, 1) ---
        __m128 x1 = _mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(r_x1, mask_23), exp_127)), ones_ps);
        __m128 y1 = _mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(r_y1, mask_23), exp_127)), ones_ps);
        __m128 x2 = _mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(r_x2, mask_23), exp_127)), ones_ps);
        __m128 y2 = _mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(r_y2, mask_23), exp_127)), ones_ps);
        __m128 x3 = _mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(r_x3, mask_23), exp_127)), ones_ps);
        __m128 y3 = _mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(r_y3, mask_23), exp_127)), ones_ps);
        __m128 x4 = _mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(r_x4, mask_23), exp_127)), ones_ps);
        __m128 y4 = _mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_and_si128(r_y4, mask_23), exp_127)), ones_ps);

        // --- Calculate distance squared (interleaved for ILP) ---
        __m128 d_sq1 = _mm_add_ps(_mm_mul_ps(x1, x1), _mm_mul_ps(y1, y1));
        __m128 d_sq2 = _mm_add_ps(_mm_mul_ps(x2, x2), _mm_mul_ps(y2, y2));
        __m128 d_sq3 = _mm_add_ps(_mm_mul_ps(x3, x3), _mm_mul_ps(y3, y3));
        __m128 d_sq4 = _mm_add_ps(_mm_mul_ps(x4, x4), _mm_mul_ps(y4, y4));

        // --- Compare and get masks ---
        int m1 = _mm_movemask_ps(_mm_cmple_ps(d_sq1, ones_ps));
        int m2 = _mm_movemask_ps(_mm_cmple_ps(d_sq2, ones_ps));
        int m3 = _mm_movemask_ps(_mm_cmple_ps(d_sq3, ones_ps));
        int m4 = _mm_movemask_ps(_mm_cmple_ps(d_sq4, ones_ps));

        // --- Accumulate hits ---
        hits += __builtin_popcount(m1) + __builtin_popcount(m2) + __builtin_popcount(m3) + __builtin_popcount(m4);
    }
    
    // --- Handle remaining tosses (scalar loop) ---
    long long remaining_tosses = tosses_per_process % 16;
    for (long long i = 0; i < remaining_tosses; ++i) {
        uint32_t r_x = xorshift32(&rng_state);
        uint32_t r_y = xorshift32(&rng_state);
        
        float x = (r_x & 0x007FFFFF) / (float)0x007FFFFF - 0.5f;
        float y = (r_y & 0x007FFFFF) / (float)0x007FFFFF - 0.5f;
        x *= 2.0f;
        y *= 2.0f;

        if (x * x + y * y <= 1.0f) {
            hits++;
        }
    }

    return hits;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 計算自己分配到的工作量
    long long int tosses_per_process = tosses_per_process = tosses / world_size + (world_rank < tosses % world_size);;
    long long int number_in_circle = compute_pi_hits_sse(tosses_per_process, world_rank);

    // TODO: binary tree reduction (每次合併一半的節點)
    for (int step = 1; step < world_size; step *= 2)
    {
        // 判斷發送者和接收者
        // 假設 world_size 為 9 (0,1,...,8)
        // step=1 時，2*step=2，奇數 rank 為發送者，偶數為接收者。(1->0, 3->2, ..., 7->6, 8不動，剩 0,2, ...,8)
        // step=2 時，2*step=4，(%4 == 0)為接收者，部分上輪接收者(0,2,...,8)變發送者。(2->0, 6->4, 8不動，剩 0,4,8)
        // step=4 時，(%8 == 0)為接收者。(4->0，剩 0,8)
        // step=8 時，(%16 == 0)為接收者。(8->0，剩 0)
        if (world_rank % (2 * step) != 0)
        {
            int dest = world_rank - step;
            MPI_Send(&number_in_circle, 1, MPI_LONG_LONG_INT, dest, 0, MPI_COMM_WORLD);
            break;
        }
        else
        {
            int source = world_rank + step;
            if (source < world_size)
            {
                long long int received_count;
                MPI_Recv(&received_count, 1, MPI_LONG_LONG_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                number_in_circle += received_count;
            }
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * number_in_circle / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
