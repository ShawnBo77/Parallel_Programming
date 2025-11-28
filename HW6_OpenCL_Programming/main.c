#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bmpfuncs.h"
#include "cycle_timer.h"
#include "helper.h"
#include "host_fe.h"
#include "serial_conv.h"

void usage(const char *progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -i  --input   <String> Input image\n");
    printf("  -f  --filter  <INT>    Use which filter (1, 2, 3)\n");
    printf("  -?  --help             This message\n");
}

int compare(const void *a, const void *b)
{
    double *x = (double *)a;
    double *y = (double *)b;
    if (*x < *y)
        return -1;
    if (*x > *y)
        return 1;
    return 0;
}

int main(int argc, char **argv)
{
    int i, j;

    // Rows and columns in the input image
    int image_height;
    int image_width;

    double start_time, end_time;

    char *input_file = "input.bmp";
    const char *output_file = "output.bmp";
    const char *ref_file = "ref.bmp";
    char *filter_file = "filter1.csv";

    // parse commandline options
    int opt;
    static struct option long_options[]
        = {{"filter", 1, 0, 'f'}, {"input", 1, 0, 'i'}, {"help", 0, 0, '?'}, {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "i:f:?", long_options, NULL)) != EOF)
    {

        switch (opt)
        {
            case 'i':
            {
                input_file = optarg;

                break;
            }
            case 'f':
            {
                int idx = atoi(optarg);
                if (idx == 2)
                    filter_file = "filter2.csv";
                else if (idx == 3)
                    filter_file = "filter3.csv";

                break;
            }
            case '?':
            default:
                usage(argv[0]);
                return 1;
        }
    }
    // end parsing of commandline options

    /*************************/
    /* OpenCL (from student) */
    /*************************/

    // Output image on the host
    float *output_image = NULL;
    double min_thread = 0;
    {
        // read filter data
        int filter_width;
        float *filter = read_filter(filter_file, &filter_width);

        // Homegrown function to read a BMP from file
        float *input_image = read_image(input_file, &image_width, &image_height);

        // Size of the input and output images on the host
        const unsigned long data_size = image_height * image_width * sizeof(float);
        output_image = (float *)malloc(data_size);

        // helper init CL
        cl_program program;
        cl_device_id device;
        cl_context context;
        init_cl(&device, &context, &program);

        double record_thread[10] = {0};
        for (int i = 0; i < 10; ++i)
        {
            memset(output_image, 0, data_size);
            start_time = current_seconds();
            // Run the host to execute the kernel
            host_fe(filter_width, filter, image_height, image_width, input_image, output_image,
                    &device, &context, &program);
            end_time = current_seconds();
            record_thread[i] = end_time - start_time;
        }
        qsort(record_thread, 10, sizeof(double), compare);
        for (int i = 3; i < 7; ++i)
        {
            min_thread += record_thread[i];
        }
        min_thread /= 4;

        printf("\n[conv opencl]:\t\t[%.3f] ms\n\n", min_thread * 1000);

        // Write the output image to file
        store_image(output_image, output_file, image_height, image_width, input_file);
        free(input_image);
        free(filter);
    }

    /**********************/
    /* Serial (reference) */
    /**********************/

    // Output image of reference on the host
    float *ref_image = NULL;
    double min_serial = 0;
    {
        // read filter data
        int filter_width;
        float *filter = read_filter(filter_file, &filter_width);

        // Homegrown function to read a BMP from file
        float *input_image = read_image(input_file, &image_width, &image_height);

        // Size of the input and output images on the host
        const unsigned long data_size = image_height * image_width * sizeof(float);
        ref_image = (float *)malloc(data_size);

        double record_serial[10] = {0};
        for (int i = 0; i < 10; ++i)
        {
            memset(ref_image, 0, data_size);
            start_time = current_seconds();
            serial_conv(filter_width, filter, image_height, image_width, input_image, ref_image);
            end_time = current_seconds();
            record_serial[i] = end_time - start_time;
        }
        qsort(record_serial, 10, sizeof(double), compare);
        for (int i = 3; i < 7; ++i)
        {
            min_serial += record_serial[i];
        }
        min_serial /= 4;

        printf("\n[conv serial]:\t\t[%.3f] ms\n\n", min_serial * 1000);

        store_image(ref_image, ref_file, image_height, image_width, input_file);
        free(input_image);
        free(filter);
    }

    int diff_counter = 0;
    for (i = 0; i < image_height; i++)
    {
        for (j = 0; j < image_width; j++)
        {
            if (fabsf(output_image[(i * image_width) + j] - ref_image[(i * image_width) + j]) > 10)
            {
                diff_counter += 1;
            }
        }
    }

    free(output_image);
    free(ref_image);

    float diff_ratio = (float)diff_counter / (float)(image_height * image_width);
    printf("Diff ratio: %f\n", diff_ratio);

    if (diff_ratio > 0.1)
    {
        printf("\n\033[31mFAILED:\tResults are incorrect!\033[0m\n");
        return -1;
    }

    printf("\n\033[32mPASS:\t(%.2fx speedup over the serial version)\033[0m\n",
           min_serial / min_thread);

    return 0;
}
