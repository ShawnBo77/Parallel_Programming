#ifndef SERIAL_CONV_H
#define SERIAL_CONV_H

void serial_conv(int filter_width,
                 const float *filter,
                 int image_height,
                 int image_width,
                 const float *input_image,
                 float *output_image);

#endif
