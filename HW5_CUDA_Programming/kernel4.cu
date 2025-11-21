#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cstring>
#include <vector>
#include <algorithm> // For std::min

#define BLOCK_X 16
#define BLOCK_Y 16

// Device function
__forceinline__ __device__
int mandel_check(float2 c, int max_iterations)
{
    float2 z = c;
    int i;

    // Periodicity Checking Variables
    float2 z_history = z; // Store a previous z value
    int check_period = 1000; // How often we check for cycles 400

    for (i = 0; i < max_iterations; ++i)
    {
        float zx2 = z.x * z.x;
        float zy2 = z.y * z.y;

        if (zx2 + zy2 > 4.0f)
            return i;

        // The core calculation
        z.y = fmaf(2.0f * z.x, z.y, c.y); // 2.0f * z.x * z.y + c.y;
        z.x = zx2 - zy2 + c.x;

        // Check if the new z is the same as our stored history z
        if (z.x == z_history.x && z.y == z_history.y) {
            return max_iterations;
        }
        
        // Every 'check_period' iterations, update the history
        if ((i & (check_period - 1)) == (check_period - 1)) { //(i + 1) % check_period == 0
             z_history = z;
        }
    }
    return max_iterations;
}

// Kernel function
__launch_bounds__(BLOCK_X * BLOCK_Y, 4)
__global__ void mandel_kernel(float lower_x,
                                    float lower_y,
                                    float step_x,
                                    float step_y,
                                    int *output,
                                    int width,
                                    int start_row,
                                    int chunk_height,
                                    int max_iterations,
                                    size_t pitch)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && local_row < chunk_height)
    {
        int global_row = start_row + local_row;
        float2 c = make_float2(lower_x + col * step_x, lower_y + global_row * step_y);
        int iterations = mandel_check(c, max_iterations);
        int *row_ptr = (int *)((char *)output + local_row * pitch);
        row_ptr[col] = iterations;
    }
}

void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img,
             int res_x,
             int res_y,
             int max_iterations)
{
    // Configuration
    const int num_streams = 8;
    const int chunk_rows = (res_y + num_streams - 1) / num_streams;
    
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    // Resource Allocation
    std::vector<cudaStream_t> streams(num_streams);
    std::vector<int*> d_outputs(num_streams, nullptr);
    std::vector<int*> h_chunks(num_streams, nullptr);
    std::vector<size_t> pitches(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
        
        int start_row = i * chunk_rows;
        if (start_row >= res_y) continue;
        int current_chunk_rows = std::min(chunk_rows, res_y - start_row);
        
        cudaMallocPitch((void **)&d_outputs[i], &pitches[i], 
                       (size_t)res_x * sizeof(int), current_chunk_rows);
                       
        cudaHostAlloc((void **)&h_chunks[i], (size_t)res_x * current_chunk_rows * sizeof(int), cudaHostAllocDefault);
    }

    // Launch all work
    dim3 threads_per_block(BLOCK_X, BLOCK_Y);
    int gridX = (res_x + BLOCK_X - 1) / BLOCK_X;
    for (int i = 0; i < num_streams; ++i)
    {
        int start_row = i * chunk_rows;
        if (start_row >= res_y) continue;
        int current_chunk_rows = std::min(chunk_rows, res_y - start_row);
        
        dim3 num_blocks(gridX, (current_chunk_rows + BLOCK_Y - 1) / BLOCK_Y);

        mandel_kernel<<<num_blocks, threads_per_block, 0, streams[i]>>>(
            lower_x, lower_y, step_x, step_y, 
            d_outputs[i], res_x, start_row, current_chunk_rows, max_iterations, pitches[i]);

        cudaMemcpy2DAsync(h_chunks[i], (size_t)res_x * sizeof(int), d_outputs[i], pitches[i],
                          (size_t)res_x * sizeof(int), current_chunk_rows, 
                          cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize and assemble results
    for (int i = 0; i < num_streams; ++i)
    {
        int start_row = i * chunk_rows;
        if (start_row >= res_y) continue;
        int current_chunk_rows = std::min(chunk_rows, res_y - start_row);

        cudaStreamSynchronize(streams[i]);

        int *dest_ptr = img + start_row * res_x;
        memcpy(dest_ptr, h_chunks[i], (size_t)res_x * current_chunk_rows * sizeof(int));
    }

    // Cleanup
    for (int i = 0; i < num_streams; ++i)
    {
        if(streams[i]) cudaStreamDestroy(streams[i]);
        if (d_outputs[i]) cudaFree(d_outputs[i]);
        if (h_chunks[i]) cudaFreeHost(h_chunks[i]);
    }
}