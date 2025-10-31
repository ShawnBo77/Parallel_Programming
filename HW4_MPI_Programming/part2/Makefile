all: matmul

matmul: main.cc matmul.cc
	mpicxx -o $@ $^ -O3 -g

clean:
	$(RM) matmul
