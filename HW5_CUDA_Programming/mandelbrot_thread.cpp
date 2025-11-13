#include "kernel.h"

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by using CUDA
void mandelbrot_thread(
    float x0, float y0, float x1, float y1, int width, int height, int max_iterations, int *output)
{
    host_fe(x1, y1, x0, y0, output, width, height, max_iterations);
}
