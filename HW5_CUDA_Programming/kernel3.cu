#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define GRID_X  40
#define GRID_Y  1

// Kernel using a grid-stride loop to process a group of pixels per thread
__global__ void mandel_kernel_grouped(float lower_x,
                                      float lower_y,
                                      float step_x,
                                      float step_y,
                                      int *output,
                                      int width,
                                      int height,
                                      int max_iterations,
                                      size_t pitch)
{
    // 計算 global thread ID (一維)
    // thread 找出在哪一個 block，block_id = blockIdx.y * gridDim.x + blockIdx.x
    // 前面的 block 含的 thread 數，block_id * (blockDim.x * blockDim.y)
    // thread 在自己的 block 中的位址，local_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y)
                    + (threadIdx.y * blockDim.x + threadIdx.x);

    // 計算 Grid 中的總 thread 數
    int total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    int total_pixels = width * height;

    // Grid-stride loop: 每個 thread 負責計算一連串的像素
    for (int pixel_idx = thread_id; pixel_idx < total_pixels; pixel_idx += total_threads)
    {
        // 將 1D index 轉回 2D
        int col = pixel_idx % width;
        int row = pixel_idx / width;

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

        int *row_ptr = (int *)((char *)output + row * pitch);
        row_ptr[col] = i;
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
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    int *h_output;
    size_t mem_size = res_x * res_y * sizeof(int);
    cudaHostAlloc((void **)&h_output, mem_size, cudaHostAllocDefault);

    int *d_output;
    size_t pitch;
    cudaMallocPitch((void **)&d_output, &pitch, res_x * sizeof(int), res_y);

    dim3 threads_per_block(BLOCK_X, BLOCK_Y);
    dim3 num_blocks(GRID_X, GRID_Y);

    mandel_kernel_grouped<<<num_blocks, threads_per_block>>>(
        lower_x, lower_y, step_x, step_y, d_output, res_x, res_y, max_iterations, pitch);

    cudaMemcpy2D(h_output, res_x * sizeof(int), d_output, pitch, res_x * sizeof(int), res_y,
                 cudaMemcpyDeviceToHost);

    memcpy(img, h_output, mem_size);

    cudaFree(d_output);
    cudaFreeHost(h_output);
}