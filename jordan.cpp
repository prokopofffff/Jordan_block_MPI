#include "jordan.h"
#include <mpi.h>

// MPI-based reduction (sum)
void MPIReduceSum(int* a, int* result, int n) {
    MPI_Allreduce(a, result, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

int Jordan(double *A, double *B, double *X, double *C, double *block, double *dop_mat, int n, int m) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double t_start = MPI_Wtime();
    double norm = A[0];

    if (rank == 0) {
        std::cout << "[MPI] Process " << rank << " Jordan() start\n";
        std::cout << "[MPI] Size " << size << "\n";
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(norm < A[i * n + j]) norm = A[i * n + j];
        }
    }

    int k = n / m;
    int l = n % m;
    int h = (l == 0 ? k : k + 1);
    int red = 0;
    // int bpt = k / size;

    for (int r = 0; r < h; r++) {
        int block_size = (r == k) ? l : m;

        if (rank == r % size) {
            get_block(A, block, n, m, r, r);
            int status = inverse(block, C, block_size, norm);
            MPI_Bcast(&status, 1, MPI_INT, r % size, MPI_COMM_WORLD);
            if (status == -1) return -1;
            set_block(A, block, n, m, r, r);
            MPI_Bcast(block, block_size * block_size, MPI_DOUBLE, r % size, MPI_COMM_WORLD);
            MPI_Bcast(C, block_size * block_size, MPI_DOUBLE, r % size, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(&red, 1, MPI_INT, r % size, MPI_COMM_WORLD); 
            if (red == -1) return -1;
            get_block(A, block, n, m, r, r); 
            MPI_Bcast(block, block_size * block_size, MPI_DOUBLE, r % size, MPI_COMM_WORLD);
            MPI_Bcast(C, block_size * block_size, MPI_DOUBLE, r % size, MPI_COMM_WORLD);
            set_block(A, block, n, m, r, r);
        }

        if (rank == r % size) {
            get_vector(B, X, n, m, r);
            multiply(C, X, dop_mat, block_size, block_size, block_size, 1);
            set_vector(B, dop_mat, n, m, r);
            MPI_Bcast(dop_mat, block_size, MPI_DOUBLE, r % size, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(dop_mat, block_size, MPI_DOUBLE, r % size, MPI_COMM_WORLD);
            set_vector(B, dop_mat, n, m, r);
        }

        for (int s = r + 1; s < h; s++) {
            int block_size_s = (s == k) ? l : m;
            if (rank == s % size) {
                get_block(A, block, n, m, r, s);
                multiply(C, block, dop_mat, block_size, block_size, block_size, block_size_s);
                set_block(A, dop_mat, n, m, r, s);
                MPI_Bcast(dop_mat, block_size * block_size_s, MPI_DOUBLE, s % size, MPI_COMM_WORLD);
            } else {
                MPI_Bcast(dop_mat, block_size * block_size_s, MPI_DOUBLE, s % size, MPI_COMM_WORLD);
                set_block(A, dop_mat, n, m, r, s);
            }
        }

        for (int i = 0; i < h; i++) {
            if (i == r) continue;
            int block_size_i = (i == k) ? l : m;
            if (rank == i % size) {
                get_block(A, C, n, m, i, r);
                for (int j = r + 1; j < h; j++) {
                    int block_size_j = (j == k) ? l : m;
                    get_block(A, block, n, m, r, j);
                    multiply(C, block, dop_mat, block_size_i, block_size, block_size, block_size_j);
                    get_block(A, block, n, m, i, j);
                    subtract(block, dop_mat, block_size_i, block_size_j);
                    set_block(A, block, n, m, i, j);
                    MPI_Bcast(block, block_size_i * block_size_j, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
                }
                get_vector(B, X, n, m, r);
                multiply(C, X, dop_mat, block_size_i, block_size, block_size, 1);
                get_vector(B, X, n, m, i);
                subtract(X, dop_mat, block_size_i, 1);
                set_vector(B, X, n, m, i);
                MPI_Bcast(X, block_size_i, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
            } else {
                for (int j = r + 1; j < h; j++) {
                    int block_size_j = (j == k) ? l : m;
                    MPI_Bcast(block, block_size_i * block_size_j, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
                    set_block(A, block, n, m, i, j);
                }
                MPI_Bcast(X, block_size_i, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
                set_vector(B, X, n, m, i);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for(int i = 0; i < n; i++){
            X[i] = B[i];
        }
    }
    MPI_Bcast(X, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    if (rank == 0) printf("[MPI] Process %d Jordan() time: %.6f seconds\n", rank, t_end - t_start);
    return 1;
}