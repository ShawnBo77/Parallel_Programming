__kernel void convolution(const int filter_width,
                          __constant float *restrict filter,
                          const int image_height,
                          const int image_width,
                          __global const float *restrict input_image,
                          __global float *restrict output_image)
{
    // Global ID
    int ix = get_global_id(0);
    int iy = get_global_id(1);

    if (ix >= image_width || iy >= image_height)
        return;

    float sum = 0.0f;
    int halo = filter_width >> 1; // filter_width / 2

    // 檢查像素邊界
    int is_safe = (ix >= halo && ix < (image_width - halo) && iy >= halo && iy < (image_height - halo));

    if (is_safe)
    {
        // 取得中心點指標
        int center_offset = iy * image_width + ix;
        __global const float *restrict img_ptr = input_image + center_offset;

        if (filter_width == 3)
        {
            // 3x3 Optimized
            // Row -1
            __global const float *row_ptr = img_ptr - image_width;
            sum += row_ptr[-1] * filter[0] + row_ptr[0] * filter[1] + row_ptr[1] * filter[2];
            // Row 0
            row_ptr = img_ptr;
            sum += row_ptr[-1] * filter[3] + row_ptr[0] * filter[4] + row_ptr[1] * filter[5];
            // Row +1
            row_ptr = img_ptr + image_width;
            sum += row_ptr[-1] * filter[6] + row_ptr[0] * filter[7] + row_ptr[1] * filter[8];
        }
        else if (filter_width == 5)
        {
            // 5x5 Optimized
            int f_idx = 0;

            // 使用指標，減少乘法
            #pragma unroll
            for (int k = -2; k <= 2; k++)
            {
                __global const float *row_ptr = img_ptr + k * image_width;
                sum += row_ptr[-2] * filter[f_idx + 0] + row_ptr[-1] * filter[f_idx + 1]
                       + row_ptr[0] * filter[f_idx + 2] + row_ptr[1] * filter[f_idx + 3]
                       + row_ptr[2] * filter[f_idx + 4];
                f_idx += 5;
            }
        }
        else if (filter_width == 7)
        {
            // 7x7 Optimized
            int f_idx = 0;
            #pragma unroll
            for (int k = -3; k <= 3; k++)
            {
                __global const float *row_ptr = img_ptr + k * image_width;
                sum += row_ptr[-3] * filter[f_idx + 0] + row_ptr[-2] * filter[f_idx + 1]
                       + row_ptr[-1] * filter[f_idx + 2] + row_ptr[0] * filter[f_idx + 3]
                       + row_ptr[1] * filter[f_idx + 4] + row_ptr[2] * filter[f_idx + 5]
                       + row_ptr[3] * filter[f_idx + 6];
                f_idx += 7;
            }
        }
        else {
            int w = image_width;
            int f_idx = 0;
            #pragma unroll
            for (int k = -halo; k <= halo; k++) {
                __global const float * row_ptr = img_ptr + k * w;
                #pragma unroll
                for (int l = -halo; l <= halo; l++) {
                    sum += row_ptr[l] * filter[f_idx++];
                }
            }
        }
    }
    else
    {
        // Border Path
        for (int k = -halo; k <= halo; k++)
        {
            int r = iy + k;
            if (r >= 0 && r < image_height)
            {
                int i_offset = r * image_width;
                int f_offset = (k + halo) * filter_width + halo;

                for (int l = -halo; l <= halo; l++)
                {
                    int c = ix + l;
                    if (c >= 0 && c < image_width)
                    {
                        sum += input_image[i_offset + c] * filter[f_offset + l];
                    }
                }
            }
        }
    }

    output_image[iy * image_width + ix] = sum;
}
