#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include "common/cycle_timer.h"
#include <emmintrin.h>

struct WorkerArgs
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
};

namespace
{

inline int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

}

extern void mandelbrot_serial(float x0,
                              float y0,
                              float x1,
                              float y1,
                              int width,
                              int height,
                              int start_row,
                              int num_rows,
                              int max_iterations,
                              int *output);

//
// worker_thread_start --
//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{

    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread could make a call to mandelbrot_serial()
    // to compute a part of the output image. For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    // Of course, you can copy mandelbrot_serial() to this file and
    // modify it to pursue a better performance.
    
    double startTime = CycleTimer::current_seconds();

    // ===============================================================
    // Method 1: Block Decomposition
    // ===============================================================
    
    // int rows_per_thread = args->height / args->numThreads;
    // int start_row = args->threadId * rows_per_thread;
    // int num_rows;

    // // The last thread handles the remaining rows
    // if (args->threadId == args->numThreads - 1) {
    //     num_rows = args->height - start_row;
    // } else {
    //     num_rows = rows_per_thread;
    // }

    // mandelbrot_serial(args->x0, args->y0, args->x1, args->y1,
    //                   args->width, args->height,
    //                   start_row, num_rows,
    //                   args->maxIterations, args->output);
    

    // ===================================================================
    // Method 2: Interleaved Decomposition
    // ===================================================================
    
    // for (unsigned int i = args->threadId; i < args->height; i += args->numThreads) {
    //     // 每次呼叫 mandelbrot_serial 只計算一行
    //     mandelbrot_serial(args->x0, args->y0, args->x1, args->y1,
    //                       args->width, args->height,
    //                       i, 1,
    //                       args->maxIterations, args->output);
    // }

    // ===================================================================
    // Method 3: Interleaved Decomposition without external function call
    // ===================================================================
    // float dx = (args->x1 - args->x0) / (float)args->width;
    // float dy = (args->y1 - args->y0) / (float)args->height;
    // const int maxIters = args->maxIterations;

    // for (unsigned int j = args->threadId; j < args->height; j += args->numThreads)
    // {
    //     for (unsigned int i = 0; i < args->width; ++i)
    //     {
    //         const float c_re = args->x0 + ((float)i * dx);
    //         const float c_im = args->y0 + ((float)j * dy);

    //         // --- mandel ---
    //         float z_re = c_re;
    //         float z_im = c_im;
            
    //         int k;
            
    //         for (k = 0; k < maxIters; ++k)
    //         {
    //             if (z_re * z_re + z_im * z_im > 4.f)
    //                 break;

    //             float new_re = (z_re * z_re) - (z_im * z_im);
    //             float new_im_prime = 2.f * z_re * z_im;
                
    //             z_re = c_re + new_re;
    //             z_im = c_im + new_im_prime;
    //         }
    //         // --------------

    //         int index = (j * args->width) + i;
    //         args->output[index] = k;
    //     }
    // }

    // ===================================================================
    // Method 4: SIMD Interleaved Decomposition
    // ===================================================================
    const int width = args->width;
    const int height = args->height;
    const int maxIters = args->maxIterations;
    const float dx = (args->x1 - args->x0) / (float)width;
    const float dy = (args->y1 - args->y0) / (float)height;

    const __m128 four_ps = _mm_set1_ps(4.0f);
    const __m128 two_ps = _mm_set1_ps(2.0f);
    const __m128i zero_si128 = _mm_setzero_si128();
    const __m128i one_epi32 = _mm_set1_epi32(1);

    for (int j = args->threadId; j < height; j += args->numThreads)
    {
        // 計算當前處理行（第 j 行）對應的 y 座標（虛部）。
        const float y = args->y0 + j * dy;
        const __m128 c_im = _mm_set1_ps(y);

        int i = 0;
        int limit = width - (width % 4); // 處理能被 4 整除的部分
        for (; i < limit; i += 4)
        {
            // 用 setr按順序填入 lane0..lane3，避免用 _mm_set_ps 會反轉順序
            __m128 c_re = _mm_setr_ps(
                args->x0 + (i + 0) * dx,
                args->x0 + (i + 1) * dx,
                args->x0 + (i + 2) * dx,
                args->x0 + (i + 3) * dx
            );

            __m128 z_re = c_re;
            __m128 z_im = c_im;
            __m128i iter = zero_si128;

            for (int k = 0; k < maxIters; k++)
            {   
                // ---------------------------------------------
                // if (z_re * z_re + z_im * z_im > 4.f) break;
                // ---------------------------------------------
                __m128 z_re2 = _mm_mul_ps(z_re, z_re);
                __m128 z_im2 = _mm_mul_ps(z_im, z_im);
                __m128 mag2 = _mm_add_ps(z_re2, z_im2);

                // _mm_cmple_ps 為 compare less than or equal
                __m128 mask = _mm_cmple_ps(mag2, four_ps); // true -> 0xFFFFFFFF
                // _mm_movemask_ps 提取每個 lane 的最高位，組成一個 4-bit interger
                int active = _mm_movemask_ps(mask);
                if (active == 0)
                    break;

                // ---------------------------------------------
                // float new_re = (z_re * z_re) - (z_im * z_im);
                // float new_im = 2.f * z_re * z_im;
                // z_re = c_re + new_re;
                // z_im = c_im + new_im;
                // ---------------------------------------------
                __m128 new_re = _mm_sub_ps(z_re2, z_im2);
                __m128 new_im = _mm_mul_ps(_mm_mul_ps(two_ps, z_re), z_im);

                z_re = _mm_add_ps(c_re, new_re);
                z_im = _mm_add_ps(c_im, new_im);

                // ---------------------------------------------
                // ---------------------------------------------
                // _mm_castps_si128 將浮點數轉成整數向量
                __m128i mask_i = _mm_castps_si128(mask);
                // mask_i & one_epi32 -> 1 (if active) or 0 (if not)
                iter = _mm_add_epi32(iter, _mm_and_si128(mask_i, one_epi32));
            }

            alignas(16) int buf[4];
            _mm_store_si128((__m128i *)buf, iter); // buf[0] 對應 lane0 即 i+0

            int base = j * width + i;
            args->output[base] = buf[0];
            args->output[base + 1] = buf[1];
            args->output[base + 2] = buf[2];
            args->output[base + 3] = buf[3];
        }

        // 剩下的部分
        for (; i < width; ++i)
        {
            float x = args->x0 + i * dx;
            args->output[j * width + i] = mandel(x, y, maxIters);
        }
    }

    double endTime = CycleTimer::current_seconds();
    printf("Thread %d took %.3f ms\n", args->threadId, (endTime - startTime) * 1000);
    // printf("Hello world from thread %d\n", args->threadId);
}

//
// mandelbrot_thread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrot_thread(int num_threads,
                       float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;

    if (num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::array<std::thread, max_threads> workers;
    std::array<WorkerArgs, max_threads> args = {};

    for (int i = 0; i < num_threads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = max_iterations;
        args[i].numThreads = num_threads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < num_threads; i++)
    {
        workers[i] = std::thread(worker_thread_start, &args[i]);
    }

    worker_thread_start(&args[0]);

    // join worker threads
    for (int i = 1; i < num_threads; i++)
    {
        workers[i].join();
    }
}
