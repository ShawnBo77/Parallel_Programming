default: conv

CC = gcc
CFLAGS = -O3 -DCL_TARGET_OPENCL_VERSION=220 -lOpenCL -m64 -ffloat-store -w -g

OBJS = $(patsubst %.c, %.o, $(wildcard *.c))

conv: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	$(RM) conv $(OBJS) output.bmp ref.bmp
