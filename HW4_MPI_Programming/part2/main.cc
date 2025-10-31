#include <fstream>
#include <iostream>
#include <mpi.h>

// *********************************************
// ** ATTENTION: YOU CANNOT MODIFY THIS FILE. **
// *********************************************

// Allocate memory for matrices a and b.
//
// n:         row number of matrix a
// m:         col number of matrix a / row number of matrix b
// l:         col number of matrix b
// a_mat:     a continuous memory placing n * m elements of int
// b_mat:     a continuous memory placing m * l elements of int
// a_mat_ptr: pointer to matrix a (could be allocated with any size and layout you want)
// b_mat_ptr: pointer to matrix b (could be allocated with any size and layout you want)
void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr);

// Just matrix multiplication.
//
// n:       row number of matrix a
// m:       col number of matrix a / row number of matrix b
// l:       col number of matrix b
// a_mat:   your allocated memory placing n * m elements of int
// b_mat:   your allocated memory placing m * l elements of int
// out_mat: a continuous memory placing n * l elements of int
void matrix_multiply(int n, int m, int l, const int *a_mat, const int *b_mat, int *out_mat);

// Remember to release your allocated memory.
void destruct_matrices(int *a_mat, int *b_mat);

int main(int argc, const char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " FILE\n";
        return 1;
    }

    int n, m, l;
    std::ifstream in(argv[1]);
    in >> n >> m >> l;

    // Owned by the main function of the rank 0 process.
    int *a_mat, *b_mat;
    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        /* Faithfully read the input matrices. */
        a_mat = new int[n * m];
        for (int y = 0; y < n; ++y)
        {
            for (int x = 0; x < m; ++x)
            {
                in >> a_mat[(y * m) + x];
            }
        }

        b_mat = new int[m * l];
        for (int y = 0; y < m; ++y)
        {
            for (int x = 0; x < l; ++x)
            {
                in >> b_mat[(x * m) + y];
            }
        }
    }

    int *student_a_mat, *student_b_mat;
    construct_matrices(n, m, l, a_mat, b_mat, &student_a_mat, &student_b_mat);

    int *out_mat;
    if (rank == 0)
    {
        out_mat = new int[n * l];
    }

    double start_time = MPI_Wtime();
    matrix_multiply(n, m, l, student_a_mat, student_b_mat, out_mat);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    destruct_matrices(student_a_mat, student_b_mat);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int y = 0; y < n; ++y)
        {
            for (int x = 0; x < l; ++x)
            {
                std::cout << out_mat[(y * l) + x] << " ";
            }
            std::cout << '\n';
        }
        std::cout << "MPI running time: " << end_time - start_time << " Seconds\n";

        delete[] out_mat;
        delete[] a_mat;
        delete[] b_mat;
    }

    MPI_Finalize();
    return 0;
}
