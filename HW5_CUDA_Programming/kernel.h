#ifndef KERNEL_H
#define KERNEL_H

void host_fe(float u_x,
             float u_y,
             float l_x,
             float l_y,
             int *image,
             int res_x,
             int res_y,
             int max_iterations);

#endif /* KERNEL_H */
