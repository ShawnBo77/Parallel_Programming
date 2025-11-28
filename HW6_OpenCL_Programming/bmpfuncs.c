#include "bmpfuncs.h"
#include <stdio.h>
#include <stdlib.h>

typedef unsigned char UChar;

void store_image(
    const float *image_out, const char *filename, int rows, int cols, const char *ref_filename)
{
    FILE *ifp = fopen(ref_filename, "rb");
    if (ifp == NULL)
    {
        perror(filename);
        exit(-1);
    }

    fseek(ifp, 10, SEEK_SET);
    int offset;
    if (fread(&offset, 4, 1, ifp) != 1)
    {
        printf("error reading header!\n");
        exit(-1);
    }

    fseek(ifp, 18, SEEK_SET);
    int height, width;
    if (fread(&width, 4, 1, ifp) != 1)
    {
        printf("error reading header!\n");
        exit(-1);
    }
    if (fread(&height, 4, 1, ifp) != 1)
    {
        printf("error reading header!\n");
        exit(-1);
    }

    fseek(ifp, 0, SEEK_SET);

    unsigned char *buffer = malloc(offset);
    if (buffer == NULL)
    {
        perror("malloc");
        exit(-1);
    }

    if (fread(buffer, 1, offset, ifp) != offset)
    {
        printf("error reading image\n");
        exit(-1);
    }

    printf("Writing output image to %s\n", filename);
    FILE *ofp = fopen(filename, "wb");
    if (ofp == NULL)
    {
        perror("opening output file");
        exit(-1);
    }
    unsigned long bytes = fwrite(buffer, 1, offset, ofp);
    if (bytes != offset)
    {
        printf("error writing header!\n");
        exit(-1);
    }

    // NOTE bmp formats store data in reverse raster order (see comment in
    // readImage function), so we need to flip it upside down here.
    int mod = width % 4;
    if (mod != 0)
    {
        mod = 4 - mod;
    }
    for (int i = height - 1; i >= 0; i--)
    {
        for (int j = 0; j < width; j++)
        {
            UChar tmp;
            tmp = (UChar)image_out[(i * cols) + j];
            fwrite(&tmp, sizeof(tmp), 1, ofp);
        }
        // In bmp format, rows must be a multiple of 4-bytes.
        // So if we're not at a multiple of 4, add junk padding.
        for (int j = 0; j < mod; j++)
        {
            UChar tmp;
            fwrite(&tmp, sizeof(tmp), 1, ofp);
        }
    }

    fclose(ofp);
    fclose(ifp);

    free(buffer);
}

/*
 * Read bmp image and convert to byte array. Also output the width and height
 */
float *read_image(const char *filename, int *width_out, int *height_out)
{
    printf("Reading input image from %s\n", filename);
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        perror(filename);
        exit(-1);
    }

    fseek(fp, 10, SEEK_SET);
    int offset;
    if (fread(&offset, 4, 1, fp) != 1)
    {
        printf("error reading header!\n");
        exit(-1);
    }

    fseek(fp, 18, SEEK_SET);
    int height, width;
    if (fread(&width, 4, 1, fp) != 1)
    {
        printf("error reading header!\n");
        exit(-1);
    }
    if (fread(&height, 4, 1, fp) != 1)
    {
        printf("error reading header!\n");
        exit(-1);
    }

    printf("width = %d\n", width);
    printf("height = %d\n", height);

    *width_out = width;
    *height_out = height;

    UChar *image_data = malloc(width * height * sizeof(UChar));
    if (image_data == NULL)
    {
        perror("malloc");
        exit(-1);
    }

    fseek(fp, offset, SEEK_SET);
    fflush(NULL);

    int mod = width % 4;
    if (mod != 0)
    {
        mod = 4 - mod;
    }

    // NOTE bitmaps are stored in upside-down raster order.  So we begin
    // reading from the bottom left pixel, then going from left-to-right,
    // read from the bottom to the top of the image.  For image analysis,
    // we want the image to be right-side up, so we'll modify it here.

    // First we read the image in upside-down

    // Read in the actual image
    for (int i = 0; i < height; i++)
    {
        // add actual data to the image
        for (int j = 0; j < width; j++)
        {
            UChar tmp;
            if (fread(&tmp, sizeof(char), 1, fp) != 1)
            {
                printf("error reading image data!\n");
                exit(-1);
            }
            image_data[(i * width) + j] = tmp;
        }
        // For the bmp format, each row has to be a multiple of 4,
        // so I need to read in the junk data and throw it away
        for (int j = 0; j < mod; j++)
        {
            UChar tmp;
            if (fread(&tmp, sizeof(char), 1, fp) != 1)
            {
                printf("error reading image data!\n");
                exit(-1);
            }
        }
    }

    // Then we flip it over
    for (int i = 0; i < height / 2; i++)
    {
        int flip_row = height - (i + 1);
        for (int j = 0; j < width; j++)
        {

            UChar tmp // NOLINT(clang-analyzer-core.uninitialized.Assign): No idea why possibly
                      // uninitialized.

                = image_data[(i * width) + j];
            image_data[(i * width) + j] = image_data[(flip_row * width) + j];
            image_data[(flip_row * width) + j] = tmp;
        }
    }

    fclose(fp);

    // Input image on the host
    float *float_image = malloc(sizeof(float) * width * height);
    if (float_image == NULL)
    {
        perror("malloc");
        exit(-1);
    }

    // Convert the BMP image to float (not required)
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float_image[(i * width) + j] = (float)image_data[(i * width) + j];
        }
    }

    free(image_data);
    return float_image;
}
