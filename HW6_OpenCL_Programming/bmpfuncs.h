#ifndef BMPFUNCS_H
#define BMPFUNCS_H

typedef unsigned char Uchar;

float *read_image(const char *filename, int *width_out, int *height_out);
void store_image(const float *image_out, const char *filename, int rows, int cols, const char *ref_filename);

#endif
