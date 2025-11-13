#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#include "cycle_timer.h"

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

extern void mandelbrot_thread(
    float x0, float y0, float x1, float y1, int width, int height, int max_iterations, int *output);

extern void mandelbrot_thread_ref(
    float x0, float y0, float x1, float y1, int width, int height, int max_iterations, int *output);

extern void
write_ppm_image(int *data, int width, int height, const char *filename, int max_iterations);

void scale_and_shift(
    float &x0, float &x1, float &y0, float &y1, float scale, float shift_x, float shift_y)
{

    x0 *= scale;
    x1 *= scale;
    y0 *= scale;
    y1 *= scale;
    x0 += shift_x;
    x1 += shift_x;
    y0 += shift_y;
    y1 += shift_y;
}

void usage(const char *progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -i  --iter <INT>       Use specified interation (>=256) (Default = 256)\n");
    printf("  -v  --view <INT>       Use specified view settings (1 or 2) (Default = 1)\n");
    printf("  -g  --gpu-only <INT>   Only run GPU or not (1 or 0) (Default = 0)\n");
    printf("  -?  --help             This message\n");
}

bool verify_result(int *gold, int *result, int width, int height)
{

    int i, j;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            if (abs(gold[(i * width) + j] - result[(i * width) + j]) > 0)
            {
                printf("Mismatch : [%d][%d], Expected : %d, Actual : %d\n", i, j,
                       gold[(i * width) + j], result[(i * width) + j]);
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char **argv)
{

    const unsigned int width = 1600;
    const unsigned int height = 1200;
    int max_iterations = 256;
    bool is_gpu_only = false;

    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays): Required by C function.
    static struct option long_options[] = {{"iter", 1, nullptr, 'i'},
                                           {"view", 1, nullptr, 'v'},
                                           {"gpu-only", 1, nullptr, 'g'},
                                           {"help", 0, nullptr, '?'},
                                           {nullptr, 0, nullptr, 0}};

    while ((opt = getopt_long(argc, argv, "i:v:g:?", long_options, nullptr)) != EOF)
    {

        switch (opt)
        {
            case 'i':
            {
                int iter = atoi(optarg);
                if (iter < 256)
                {
                    fprintf(stderr, "Iteration should >= 256\n");
                    return 1;
                }
                max_iterations = iter;
                break;
            }
            case 'v':
            {
                int view_index = atoi(optarg);
                // change view settings
                if (view_index == 2)
                {
                    float scale_value = .015f;
                    float shift_x = -.986f;
                    float shift_y = .30f;
                    scale_and_shift(x0, x1, y0, y1, scale_value, shift_x, shift_y);
                }
                else if (view_index > 1)
                {
                    fprintf(stderr, "Invalid view index\n");
                    return 1;
                }
                break;
            }
            case 'g':
            {
                int flag = atoi(optarg);
                // change GPU settings
                if (flag == 1 || flag == 0)
                {
                    is_gpu_only = flag;
                }
                else
                {
                    fprintf(stderr, "Invalid setting. Only allow 0 or 1.\n");
                    return 1;
                }
                break;
            }
            case '?':
            default:
                usage(argv[0]);
                return 1;
        }
    }
    // end parsing of commandline options

    int *output_test = new int[width * height];
    int *output_thread = new int[width * height];

    //
    // Run the serial implementation.  Run the code three times and
    // take the minimum to get a good estimate.
    //
    double min_serial = 1e30;
    if (!is_gpu_only)
    {
        for (int i = 0; i < 5; ++i)
        {
            memset(output_test, 0, width * height * sizeof(int));
            double start_time = CycleTimer::current_seconds();
            mandelbrot_serial(x0, y0, x1, y1, width, height, 0, height, max_iterations,
                              output_test);
            double end_time = CycleTimer::current_seconds();
            min_serial = std::min(min_serial, end_time - start_time);
        }

        printf("[mandelbrot serial]:\t\t[%.3f] ms\n", min_serial * 1000);
    }

    //
    // Run the reference threaded version
    //

    double min_ref = 0;
    std::array<double, 10> record_ref = {0};
    for (int i = 0; i < 10; ++i)
    {
        memset(output_test, 0, width * height * sizeof(int));
        double start_time = CycleTimer::current_seconds();
        mandelbrot_thread_ref(x0, y0, x1, y1, width, height, max_iterations, output_test);
        double end_time = CycleTimer::current_seconds();
        record_ref[i] = end_time - start_time;
    }
    std::sort(record_ref.begin(), record_ref.end());
    for (int i = 3; i < 7; ++i)
    {
        min_ref += record_ref[i];
    }
    min_ref /= 4;

    printf("[mandelbrot reference]:\t\t[%.3f] ms\n", min_ref * 1000);
    write_ppm_image(output_test, width, height, "mandelbrot-ref.ppm", max_iterations);

    //
    // Run the threaded version
    //

    double min_thread = 0;
    std::array<double, 10> record_thread = {0};
    for (int i = 0; i < 10; ++i)
    {
        memset(output_thread, 0, width * height * sizeof(int));
        double start_time = CycleTimer::current_seconds();
        mandelbrot_thread(x0, y0, x1, y1, width, height, max_iterations, output_thread);
        double end_time = CycleTimer::current_seconds();
        record_thread[i] = end_time - start_time;
    }
    std::sort(record_thread.begin(), record_thread.end());
    for (int i = 3; i < 7; ++i)
    {
        min_thread += record_thread[i];
    }
    min_thread /= 4;

    printf("[mandelbrot thread]:\t\t[%.3f] ms\n", min_thread * 1000);
    write_ppm_image(output_thread, width, height, "mandelbrot-thread.ppm", max_iterations);

    if (!verify_result(output_test, output_thread, width, height))
    {
        printf("Error : Output from threads does not match test output\n");

        delete[] output_test;
        delete[] output_thread;

        return 1;
    }

    // compute speedup
    if (!is_gpu_only)
        printf("\t\t\t\t(%.2fx speedup over the CPU serial version)\n", min_serial / min_thread);
    printf("\t\t\t\t(%.2fx speedup over the reference)\n", min_ref / min_thread);

    delete[] output_test;
    delete[] output_thread;

    return 0;
}
