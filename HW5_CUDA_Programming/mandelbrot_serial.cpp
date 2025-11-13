namespace
{

int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

} // namespace

//
// MandelbrotSerial --
//
// Compute an image visualizing the mandelbrot set.  The resulting
// array contains the number of iterations required before the complex
// number corresponding to a pixel could be rejected from the set.
//
// * x0, y0, x1, y1 describe the complex coordinates mapping
//   into the image viewport.
// * width, height describe the size of the output image
// * startRow, totalRows describe how much of the image to compute
void mandelbrot_serial(float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int start_row,
                       int total_rows,
                       int max_iterations,
                       int *output)
{
    float dx = (x1 - x0) / (float)width;
    float dy = (y1 - y0) / (float)height;

    int end_row = start_row + total_rows;

    for (int j = start_row; j < end_row; j++)
    {
        for (int i = 0; i < width; ++i)
        {
            float x = x0 + ((float)i * dx);
            float y = y0 + ((float)j * dy);

            int index = ((j * width) + i);
            output[index] = mandel(x, y, max_iterations);
        }
    }
}
