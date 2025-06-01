#include "jordan.h"
#include <mpi.h>

int Jordan(double *A, double *B, double *X_output, int n, int m, double norm_val) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int k_div = n / m;
    int l_rem = n % m;
    int h_blocks = (l_rem == 0) ? k_div : k_div + 1;

    double *C_inverse_diag = new double[m * m];
    double *temp_A_block = new double[m * m];
    double *temp_product_matrix = new double[m * m];
    double *temp_B_segment = new double[m];
    double *A_ir_block_buffer = new double[m*m];

    for (int r = 0; r < h_blocks; ++r) {
        int r_block_dim = (r == k_div && l_rem != 0) ? l_rem : m;
        int diag_owner_rank = r % size;
        int inv_status = 1;

        if (rank == diag_owner_rank) {
            get_block(A, temp_A_block, n, m, r, r);
            inv_status = inverse(temp_A_block, C_inverse_diag, r_block_dim, norm_val);
            set_block(A, temp_A_block, n, m, r, r);
        }
        MPI_Bcast(&inv_status, 1, MPI_INT, diag_owner_rank, MPI_COMM_WORLD);

        if (inv_status == -1) {
            delete[] C_inverse_diag; delete[] temp_A_block; delete[] temp_product_matrix; delete[] temp_B_segment; delete[] A_ir_block_buffer;
            if (rank == 0) fprintf(stderr, "Matrix inversion failed at block %d.\n", r);
            return -1;
        }

        MPI_Bcast(temp_A_block, r_block_dim * r_block_dim, MPI_DOUBLE, diag_owner_rank, MPI_COMM_WORLD);
        if (rank != diag_owner_rank) {
            set_block(A, temp_A_block, n, m, r, r);
        }
        MPI_Bcast(C_inverse_diag, r_block_dim * r_block_dim, MPI_DOUBLE, diag_owner_rank, MPI_COMM_WORLD);

        if (rank == diag_owner_rank) {
            get_vector(B, temp_B_segment, n, m, r);
            multiply(C_inverse_diag, temp_B_segment, temp_product_matrix, r_block_dim, r_block_dim, r_block_dim, 1);
            set_vector(B, temp_product_matrix, n, m, r);
        }
        MPI_Bcast(B + r * m, r_block_dim, MPI_DOUBLE, diag_owner_rank, MPI_COMM_WORLD);

        for (int s_col_task_idx = rank; s_col_task_idx < (h_blocks - (r + 1)); s_col_task_idx += size) {
            int s = r + 1 + s_col_task_idx;
            if (s >= h_blocks) continue;
            int s_block_dim = (s == k_div && l_rem != 0) ? l_rem : m;

            get_block(A, temp_A_block, n, m, r, s);
            multiply(C_inverse_diag, temp_A_block, temp_product_matrix, r_block_dim, r_block_dim, r_block_dim, s_block_dim);
            set_block(A, temp_product_matrix, n, m, r, s);
        }

        for (int s_bcast = r + 1; s_bcast < h_blocks; ++s_bcast) {
            int s_block_dim_bcast = (s_bcast == k_div && l_rem != 0) ? l_rem : m;
            int s_col_task_idx_bcast = s_bcast - (r + 1);
            int ars_owner_rank = s_col_task_idx_bcast % size;

            if (rank == ars_owner_rank) {
                get_block(A, temp_A_block, n, m, r, s_bcast);
            }
            MPI_Bcast(temp_A_block, r_block_dim * s_block_dim_bcast, MPI_DOUBLE, ars_owner_rank, MPI_COMM_WORLD);
            if (rank != ars_owner_rank) {
                set_block(A, temp_A_block, n, m, r, s_bcast);
            }
        }

        for (int i = 0; i < h_blocks; ++i) {
            if (i == r) continue;
            int i_block_dim = (i == k_div && l_rem != 0) ? l_rem : m;

            get_block(A, A_ir_block_buffer, n, m, i, r);

            for (int j_col_task_idx = rank; j_col_task_idx < (h_blocks - (r + 1)); j_col_task_idx += size) {
                int j = r + 1 + j_col_task_idx;
                if (j >= h_blocks) continue;
                int j_block_dim = (j == k_div && l_rem != 0) ? l_rem : m;

                get_block(A, temp_A_block, n, m, r, j);
                multiply(A_ir_block_buffer, temp_A_block, temp_product_matrix, i_block_dim, r_block_dim, r_block_dim, j_block_dim);

                get_block(A, temp_A_block, n, m, i, j);
                subtract(temp_A_block, temp_product_matrix, i_block_dim, j_block_dim);
                set_block(A, temp_A_block, n, m, i, j);
            }

            for (int j_bcast = r + 1; j_bcast < h_blocks; ++j_bcast) {
                int j_block_dim_bcast = (j_bcast == k_div && l_rem != 0) ? l_rem : m;
                int j_col_task_idx_bcast = j_bcast - (r+1);
                int aij_owner_rank = j_col_task_idx_bcast % size;

                if (rank == aij_owner_rank) {
                    get_block(A, temp_A_block, n, m, i, j_bcast);
                }
                MPI_Bcast(temp_A_block, i_block_dim * j_block_dim_bcast, MPI_DOUBLE, aij_owner_rank, MPI_COMM_WORLD);
                if (rank != aij_owner_rank) {
                    set_block(A, temp_A_block, n, m, i, j_bcast);
                }
            }

            int bi_owner_rank;
            int num_active_procs_for_Bi;
            if (i < r) {
                num_active_procs_for_Bi = std::min(size, r);
                if (num_active_procs_for_Bi > 0) {
                    bi_owner_rank = i % num_active_procs_for_Bi;
                } else {
                     bi_owner_rank = 0;
                }
            } else {
                num_active_procs_for_Bi = std::min(size, h_blocks - (r+1));
                if (num_active_procs_for_Bi > 0) {
                    bi_owner_rank = (i - (r+1)) % num_active_procs_for_Bi;
                } else {
                    bi_owner_rank = 0;
                }
            }
            if ( (i < r && r == 0) || (i > r && h_blocks - (r+1) == 0) ) {
                bi_owner_rank = 0;
            }


            if (rank == bi_owner_rank) {
                get_vector(B, temp_B_segment, n, m, r);
                multiply(A_ir_block_buffer, temp_B_segment, temp_product_matrix, i_block_dim, r_block_dim, r_block_dim, 1);

                get_vector(B, temp_B_segment, n, m, i);
                subtract(temp_B_segment, temp_product_matrix, i_block_dim, 1);
                set_vector(B, temp_B_segment, n, m, i);
            }
            MPI_Bcast(B + i * m, i_block_dim, MPI_DOUBLE, bi_owner_rank, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for(int idx = 0; idx < n; ++idx){
            X_output[idx] = B[idx];
        }
    }
    MPI_Bcast(X_output, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] C_inverse_diag;
    delete[] temp_A_block;
    delete[] temp_product_matrix;
    delete[] temp_B_segment;
    delete[] A_ir_block_buffer;

    return 1;
}
