CC = gcc
CFLAGS = -O3 -Wall -pthread -mavx2 -ffast-math
TARGET = pi.out

all: $(TARGET)

# Rule to build the executable
$(TARGET): pi.c
	$(CC) $(CFLAGS) -o $(TARGET) pi.c

# Rule to clean up build files
.PHONY: clean
clean:
	rm -f $(TARGET)
