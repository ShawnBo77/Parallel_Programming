#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define BLOCK_X 16
#define BLOCK_Y 16

// CUDA kernel to compute Mandelbrot set
// __global__ 告訴 CUDA 編譯器，這是 Kernel 函式，將從 CPU 被呼叫，在 GPU 上執行
__global__ void mandel_kernel(float lower_x,
                              float lower_y,
                              float step_x,
                              float step_y,
                              int *output,
                              int width,
                              int height,
                              int max_iterations)
{
    // Calculate the unique global thread ID
    // blockIdx.x, blockIdx.y: 目前 thread 所在的 Block 在 Grid 中的 2D 索引。
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image bounds
    if (col < width && row < height)
    {
        float c_re = lower_x + col * step_x;
        float c_im = lower_y + row * step_y;

        float z_re = c_re;
        float z_im = c_im;

        int i;
        for (i = 0; i < max_iterations; ++i)
        {
            if (z_re * z_re + z_im * z_im > 4.0f)
                break;

            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.0f * z_re * z_im;

            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }

        // Write the result to the output array
        int index = row * width + col;
        output[index] = i;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img,
             int res_x,
             int res_y,
             int max_iterations)
{
    // Calculate step sizes for mapping pixels to the complex plane
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    // Allocate host memory using new.
    int *h_output = new int[res_x * res_y];

    // Allocate device memory using cudaMalloc.
    int *d_output;
    size_t mem_size = res_x * res_y * sizeof(int);
    cudaMalloc((void **)&d_output, mem_size);

    // Set up the grid and block dimensions for the kernel launch.
    dim3 threads_per_block(BLOCK_X, BLOCK_Y);
    dim3 num_blocks((res_x + BLOCK_X - 1) / BLOCK_X, (res_y + BLOCK_Y - 1) / BLOCK_Y);

    // Launch the kernel.
    // <<<num_blocks, threads_per_block>>> 是 CUDA 語法，用來啟動 Kernel
    mandel_kernel<<<num_blocks, threads_per_block>>>(lower_x, lower_y, step_x, step_y, d_output,
                                                     res_x, res_y, max_iterations);

    // Copy the results from device memory back to host memory.
    cudaMemcpy(h_output, d_output, mem_size, cudaMemcpyDeviceToHost);

    memcpy(img, h_output, mem_size);

    cudaFree(d_output);
    delete[] h_output;
}