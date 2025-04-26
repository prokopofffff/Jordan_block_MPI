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
    rank++;

    std::cout << "[MPI] Process " << rank << " Jordan() start\n";
    std::cout << "[MPI] Size" << size << "\n";

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(norm < A[i * n + j]) norm = A[i * n + j];
        }
    }

    int k = n / m;
    int l = n % m;
    int h = (l == 0 ? k : k + 1);
    int red = 0;
    int bpt = k / size;

    for (int r = 0; r < h; r++) {
        int kp = (h - r - 1) / size;
        int lp = (h - r - 1) % size;
        int is_last_k = (r == k);
        int block_size = is_last_k ? l : m;

        get_block(A, block, n, m, r, r);
        std::cout << "block before inverse\n";
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                std::cout << A[i * n + j] << " ";

            }
            std::cout << std::endl;

        }
        int status = inverse(block, C, block_size, norm);
        std::cout << "block after inverse\n";
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                std::cout << C[i * n + j] << " ";

            }
            std::cout << std::endl;

        }
        MPIReduceSum(&status, &status, 1);
        if (status == -1) return -1;
        set_block(A, block, n, m, r, r);

        for (int x = 0; x < kp; x++) {
            int s = r + x * size + rank;
            int is_last_s = (s == k);
            int block_size_s = is_last_s ? l : m;

            get_block(A, block, n, m, r, s);
            multiply(C, block, dop_mat, block_size, block_size, block_size, block_size_s);
            set_block(A, dop_mat, n, m, r, s);
        }

        std::cout << "block after multiplication\n";
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                std::cout << A[i * n + j] << " ";

            }
            std::cout << std::endl;

        }

        if (rank <= lp) {
            int s = r + kp * size + rank;
            int is_last_s = (s == k);
            int block_size_s = is_last_s ? l : m;

            get_block(A, block, n, m, r, s);
            multiply(C, block, dop_mat, block_size, block_size, block_size, block_size_s);
            set_block(A, dop_mat, n, m, r, s);
        }

        std::cout << "block after dop multiplication\n";
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                std::cout << A[i * n + j] << " ";

            }
            std::cout << std::endl;

        }

        if (rank == size) {
            get_vector(B, X, n, m, r);
            multiply(C, X, dop_mat, block_size, block_size, block_size, 1);
            set_vector(B, dop_mat, n, m, r);
        }

        std::cout << "block after vector multiplication\n";
        for(int i = 0; i < n; i++){
            std::cout << B[i] << " ";
            std::cout << std::endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        for(int j = 0; j <= bpt; j++){
            int i = j * size + rank - 1;
            MPIReduceSum(&red, &red, 1);
            if(i >= r) i++;
            if(i >= h) break;
            int block_size_rows = (i == k) ? l : m;

            get_block(A, C, n, m, i, r);

            for(int c = r + 1; c < h; c++){
                int block_size_cols = (c == k) ? l : m;

                get_block(A, block, n, m, r, c);
                multiply(C, block, dop_mat, block_size_rows, m, m, block_size_cols);
                get_block(A, block, n, m, i, c);
                subtract(block, dop_mat, block_size_rows, block_size_cols);
                set_block(A, block, n, m, i, c);
            }

            std::cout << "block after sub\n";
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    std::cout << A[i * n + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << j << "!!!!!!!!!!!!!\n";
            get_vector(B, X, n, m, r);
            std::cout << "vector before multiply\n";
            for(int i = 0; i < n; i++){
                std::cout << X[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "block CCCCCCCCCCC\n";
            for(int i = 0; i < n; i++){
                for(int j = 0; j < m; j++){
                    std::cout << C[i * n + j] << " ";
                }
                std::cout << std::endl;
            }
            multiply(C, X, dop_mat, block_size_rows, block_size, block_size, 1);
            std::cout << "vector after multiply\n";
            for(int i = 0; i < n; i++){
                std::cout << dop_mat[i] << " ";
            }
            std::cout << std::endl;
            get_vector(B, X, n, m, i);
            std::cout << "vector before sub\n";
            for(int i = 0; i < n; i++){
                std::cout << X[i] << " ";
            }
            std::cout << std::endl;
            subtract(X, dop_mat, block_size_rows, 1);
            std::cout << "vector after sub\n";
            for(int i = 0; i < n; i++){
                std::cout << X[i] << " ";
            }
            std::cout << std::endl;
            set_vector(B, X, n, m, i);
            std::cout << "vector after set\n";
            for(int i = 0; i < n; i++){
                std::cout << B[i] << " ";
            }
            std::cout << std::endl;
        }
        MPIReduceSum(&red, &red, 1);
    }

    if (rank == 0) {
        for(int i = 0; i < n; i++){
            X[i] = B[i];
        }
    }

    double t_end = MPI_Wtime();
    printf("[MPI] Process %d Jordan() time: %.6f seconds\n", rank, t_end - t_start);
    return 1;
}