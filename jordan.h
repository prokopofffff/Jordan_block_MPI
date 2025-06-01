#ifndef JORDAN
#define JORDAN

#include "funcs.h"
#include <iostream>
#include <mpi.h>
#include "matrix.h"
#include <sys/resource.h>
#ifdef __linux__
#include <sys/sysinfo.h>
#elif __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#include <sys/time.h>
#include <time.h>

struct mpi_args{
    double* A;
    double* B;
    double* C;
    double* dop_mat;
    double* block;
    double* X;
    int n;
    int m;
    int p;
    double norm;
    int status;
    int rank;
    int size;
};

int Jordan(double *A, double *B, double *X_output, int n, int m, double norm_val);

#endif
