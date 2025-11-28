#include "serial_conv.h"

void serial_conv(int filter_width,
                 const float *filter,
                 int image_height,
                 int image_width,
                 const float *input_image,
                 float *output_image)
{
    // Iterate over the rows of the source image
    int halffilter_size = filter_width / 2;
    float sum;
    int i, j, k, l;

    for (i = 0; i < image_height; i++)
    {
        // Iterate over the columns of the source image
        for (j = 0; j < image_width; j++)
        {
            sum = 0; // Reset sum for new source pixel
            // Apply the filter to the neighborhood
            for (k = -halffilter_size; k <= halffilter_size; k++)
            {
                for (l = -halffilter_size; l <= halffilter_size; l++)
                {
                    if (i + k >= 0 && i + k < image_height && j + l >= 0 && j + l < image_width)
                    {
                        sum += input_image[((i + k) * image_width) + j + l]
                               * filter[((k + halffilter_size) * filter_width) + l
                                        + halffilter_size];
                    }
                }
            }
            output_image[(i * image_width) + j] = sum;
        }
    }
}
