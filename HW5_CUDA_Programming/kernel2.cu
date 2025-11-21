#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define BLOCK_X 16
#define BLOCK_Y 16

// CUDA kernel to compute Mandelbrot set using pitched memory
// __global__ 告訴 CUDA 編譯器，這是 Kernel 函式，將從 CPU 被呼叫，在 GPU 上執行
__global__ void mandel_kernel_pitch(float lower_x,
                                    float lower_y,
                                    float step_x,
                                    float step_y,
                                    int *output,
                                    int width,
                                    int height,
                                    int max_iterations,
                                    size_t pitch)
{
    // 計算 global thread ID
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

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

        // 算記憶體位置
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

    // Allocate host memory using cudaHostAlloc (Pinned Memory).
    // 作業系統不會移動這塊記憶體，實體位址固定 DMA 就可直接存取，無需透過臨時緩衝區
    // cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags)
    // Allocates size bytes of host memory that is page-locked and accessible to the device.
    int *h_output;
    size_t mem_size = res_x * res_y * sizeof(int);
    cudaHostAlloc((void **)&h_output, mem_size, cudaHostAllocDefault);

    // Allocate device memory using cudaMallocPitch.
    // 對齊記憶體存取位址，計算一個間距 (pitch)，確保下一行的起始位址落在的對齊邊界上
    // cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
    // Allocates at least width (in bytes) * height bytes of linear memory on the device
    // and returns in *devPtr a pointer to the allocated memory.
    int *d_output; // 配置好的記憶體位址
    size_t pitch;  // GPU 記憶體中每行的寬度 (in bytes)
    cudaMallocPitch((void **)&d_output, &pitch, res_x * sizeof(int), res_y);

    // Set up the grid and block dimensions for the kernel launch.
    dim3 threads_per_block(BLOCK_X, BLOCK_Y);
    dim3 num_blocks((res_x + BLOCK_X - 1) / BLOCK_X, (res_y + BLOCK_Y - 1) / BLOCK_Y);

    // Launch the kernel, passing the pitch value.
    mandel_kernel_pitch<<<num_blocks, threads_per_block>>>(
        lower_x, lower_y, step_x, step_y, d_output, res_x, res_y, max_iterations, pitch);

    // Copy the results using cudaMemcpy2D for pitched memory.
    // cudaError_t cudaMemcpy2D(void* dst,
    //                          size_t dpitch,
    //                          const void* src,
    //                          size_t spitch,
    //                          size_t width,
    //                          size_t height,
    //                          enum cudaMemcpyKind kind)
    // Copies a matrix (height rows of width bytes each) from the memory area
    // pointed to by src to the memory area pointed to by dst
    cudaMemcpy2D(h_output, res_x * sizeof(int), d_output, pitch, res_x * sizeof(int), res_y,
                 cudaMemcpyDeviceToHost);

    // Copy to final output buffer
    memcpy(img, h_output, mem_size);

    cudaFree(d_output);
    cudaFreeHost(h_output); // 使用 cudaFreeHost 來釋放 Pinned Memory
}