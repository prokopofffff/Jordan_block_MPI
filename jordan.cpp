#include "jordan.h"

// MPI-based reduction (sum)
void MPIReduceSum(int* a, int* result, int n) {
    MPI_Allreduce(a, result, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

// New parallel Jordan function
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

    // Find norm (max element) for stability
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(norm < A[i * n + j]) norm = A[i * n + j];
        }
    }

    int k = n / m;
    int l = n % m;
    int h = (l == 0 ? k : k + 1);

    // Temporary buffers for block communication
    double *block_buf = new double[m * m];
    double *C_buf = new double[m * m];
    double *dop_buf = new double[m * m];
    double *X_buf = new double[m];

    for (int r = 0; r < h; r++) {
        int block_size = (r == k) ? l : m;
        int diag_owner = r % size;
        int status = 1;
        int num_blocks = h; // number of blocks in this row/column
        int active_procs = std::min(size, num_blocks);

        // 1. Invert diagonal block (only one process does this)
        if (rank == diag_owner) {
            get_block(A, block, n, m, r, r);
            status = inverse(block, C, block_size, norm);
            set_block(A, block, n, m, r, r);
        }
        MPI_Bcast(&status, 1, MPI_INT, diag_owner, MPI_COMM_WORLD);
        if (status == -1) {
            delete[] block_buf; delete[] C_buf; delete[] dop_buf; delete[] X_buf;
            return -1;
        }
        MPI_Bcast(block, block_size * block_size, MPI_DOUBLE, diag_owner, MPI_COMM_WORLD);
        MPI_Bcast(C, block_size * block_size, MPI_DOUBLE, diag_owner, MPI_COMM_WORLD);

        // 2. Update B vector (parallel over all processes)
        for (int i = rank; i < num_blocks; i += size) {
            if (i == r) {
                get_vector(B, X, n, m, r);
                multiply(C, X, dop_mat, block_size, block_size, block_size, 1);
                set_vector(B, dop_mat, n, m, r);
            }
        }
        MPI_Bcast(B + r * m, block_size, MPI_DOUBLE, diag_owner, MPI_COMM_WORLD);

        // 3. Update row blocks to the right of the diagonal (parallel)
        int num_row_blocks = h - (r + 1);
        int row_active_procs = std::min(size, num_row_blocks);
        for (int s = r + 1 + rank; s < h; s += size) {
            int block_size_s = (s == k) ? l : m;
            get_block(A, block, n, m, r, s);
            multiply(C, block, dop_mat, block_size, block_size, block_size, block_size_s);
            set_block(A, dop_mat, n, m, r, s);
        }
        for (int s = r + 1; s < h; s++) {
            int block_size_s = (s == k) ? l : m;
            int block_index = s - (r + 1);
            int owner = block_index % row_active_procs;
            get_block(A, block_buf, n, m, r, s);
            MPI_Bcast(block_buf, block_size * block_size_s, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            set_block(A, block_buf, n, m, r, s);
        }

        // 4. Update all other blocks and B (parallel)
        for (int i = 0; i < h; i++) {
            if (i == r) continue;
            int block_size_i = (i == k) ? l : m;
            int num_col_blocks = h - (r + 1);
            int col_active_procs = std::min(size, num_col_blocks);
            // Update row blocks
            for (int j = r + 1 + rank; j < h; j += size) {
                int block_size_j = (j == k) ? l : m;
                get_block(A, C_buf, n, m, i, r);
                get_block(A, block, n, m, r, j);
                multiply(C_buf, block, dop_mat, block_size_i, block_size, block_size, block_size_j);
                get_block(A, block, n, m, i, j);
                subtract(block, dop_mat, block_size_i, block_size_j);
                set_block(A, block, n, m, i, j);
            }
            for (int j = r + 1; j < h; j++) {
                int block_size_j = (j == k) ? l : m;
                int block_index = j - (r + 1);
                int owner = block_index % col_active_procs;
                get_block(A, block_buf, n, m, i, j);
                MPI_Bcast(block_buf, block_size_i * block_size_j, MPI_DOUBLE, owner, MPI_COMM_WORLD);
                set_block(A, block_buf, n, m, i, j);
            }
            // Update B
            int bcast_owner = (i - (r + 1));
            int bcast_active_procs = std::min(size, h - (r + 1));
            bcast_owner = (bcast_owner >= 0) ? (bcast_owner % bcast_active_procs) : 0;
            if ((i - (r + 1)) >= 0 && ((i - (r + 1)) % bcast_active_procs == rank)) {
                get_block(A, C_buf, n, m, i, r);
                get_vector(B, X_buf, n, m, r);
                multiply(C_buf, X_buf, dop_buf, block_size_i, block_size, block_size, 1);
                get_vector(B, X_buf, n, m, i);
                subtract(X_buf, dop_buf, block_size_i, 1);
                set_vector(B, X_buf, n, m, i);
            }
            MPI_Bcast(B + i * m, block_size_i, MPI_DOUBLE, bcast_owner, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Copy solution
    if (rank == 0) {
        for(int i = 0; i < n; i++){
            X[i] = B[i];
        }
    }
    MPI_Bcast(X, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    if (rank == 0) printf("[MPI] Process %d Jordan() time: %.6f seconds\n", rank, t_end - t_start);
    delete[] block_buf; delete[] C_buf; delete[] dop_buf; delete[] X_buf;
    return 1;
}
