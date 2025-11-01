#include <algorithm>
#include <mpi.h>
#include <vector>

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    /* TODO: The data is stored in a_mat and b_mat.
     * You need to allocate memory for a_mat_ptr and b_mat_ptr,
     * and copy the data from a_mat and b_mat to a_mat_ptr and b_mat_ptr, respectively.
     * You can use any size and layout you want if they provide better performance.
     * Unambitiously copying the data is also acceptable.
     *
     * The matrix multiplication will be performed on a_mat_ptr and b_mat_ptr.
     */
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Calculate workload distribution for matrix A
    int rows_per_proc = n / world_size;
    int remainder_rows = n % world_size;
    // `counts` stores how many rows each process will get.
    // `displs` stores the displacement (offset) of rows for each process in the original matrix A.
    std::vector<int> counts(world_size), displs(world_size);
    int offset = 0;
    for (int i = 0; i < world_size; i++)
    {
        counts[i] = (i < remainder_rows) ? rows_per_proc + 1 : rows_per_proc;
        displs[i] = offset;
        offset += counts[i];
    }

    // 用 MPI_Scatterv 把 a_mat 從 Rank 0 分給其他 process
    int local_n = counts[rank];
    int *local_a = local_n > 0 ? new int[local_n * m] : NULL;
    if (rank == 0)
    {
        // Convert row counts to element counts for MPI_Scatterv(收元素個數)
        std::vector<int> send_counts(world_size), send_displs(world_size);
        for (int i = 0; i < world_size; i++)
        {
            send_counts[i] = counts[i] * m;
            send_displs[i] = displs[i] * m;
        }
        // int MPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[],
        //              MPI_Datatype sendtype, void *recvbuf,
        //              int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
        MPI_Scatterv(a_mat, send_counts.data(), send_displs.data(), MPI_INT, local_a, local_n * m,
                     MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, local_a, local_n * m, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Rank 0 廣播 b_mat，存到其他 process 的 local_b
    int *local_b = new int[m * l];
    if (rank == 0)
    {
        int *b_trans = new int[m * l];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < l; j++)
                b_trans[i * l + j] = b_mat[j * m + i];
        // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm )
        MPI_Bcast(b_trans, m * l, MPI_INT, 0, MPI_COMM_WORLD);
        std::copy(b_trans, b_trans + m * l, local_b);
        delete[] b_trans;
    }
    else
    {
        MPI_Bcast(local_b, m * l, MPI_INT, 0, MPI_COMM_WORLD);
    }

    *a_mat_ptr = local_a;
    *b_mat_ptr = local_b;
}

void matrix_multiply(const int n,
                     const int m,
                     const int l,
                     const int *__restrict a_mat,
                     const int *__restrict b_mat,
                     int *__restrict out_mat)
{
    /* TODO: Perform matrix multiplication on a_mat and b_mat. Which are the matrices you've
     * constructed. The result should be stored in out_mat, which is a continuous memory placing n *
     * l elements of int. You need to make sure rank 0 receives the result.
     */
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rows_per_proc = n / world_size;
    int remainder_rows = n % world_size;
    int local_n = (rank < remainder_rows) ? rows_per_proc + 1 : rows_per_proc;
    int *local_c = new int[local_n * l];
    std::fill(local_c, local_c + local_n * l, 0);

    // matrix multiplication
    const int BLOCK = 128;
    for (int ii = 0; ii < local_n; ii += BLOCK)
    {
        for (int kk = 0; kk < m; kk += BLOCK)
        {
            for (int jj = 0; jj < l; jj += BLOCK)
            {
                int i_max = std::min(ii + BLOCK, local_n);
                int k_max = std::min(kk + BLOCK, m);
                int j_max = std::min(jj + BLOCK, l);
                for (int i = ii; i < i_max; i++)
                {
                    for (int k = kk; k < k_max; k++)
                    {
                        int a_val = a_mat[i * m + k];
                        for (int j = jj; j < j_max; j++)
                        {
                            local_c[i * l + j] += a_val * b_mat[k * l + j]; // row-major
                        }
                    }
                }
            }
        }
    }

    // Gather local results (local_c) to Rank 0
    // Counts and displacements for the final matrix C
    if (rank == 0)
    {
        std::vector<int> recv_counts(world_size), recv_displs(world_size);
        int offset = 0;
        for (int i = 0; i < world_size; i++)
        {
            int rows = (i < remainder_rows) ? rows_per_proc + 1 : rows_per_proc;
            recv_counts[i] = rows * l;
            recv_displs[i] = offset;
            offset += recv_counts[i];
        }
        // int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        //             void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype
        //             recvtype, int root, MPI_Comm comm)
        MPI_Gatherv(local_c, local_n * l, MPI_INT, out_mat, recv_counts.data(), recv_displs.data(),
                    MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gatherv(local_c, local_n * l, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }

    delete[] local_c;
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    /* TODO */
    delete[] a_mat;
    delete[] b_mat;
}
