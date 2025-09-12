/**
 * @file    mlwf.c
 * @brief   This file contains functions to generate wannier inputs.
 *
 * @author  Zehan Liu <2311531808@qq.com>
 * Copyright (c) 2025
 */

#include "mlwf.h"

#include <mpi.h>

void Generate_Wannier_Inputs(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1, t2;
    if (!rank)
        printf("\nStart generating wannier inputs ... \n");
    t1 = MPI_Wtime();

    // generate wannier inputs
    Calculate_MMN(pSPARC);

    Calculate_AMN(pSPARC);

    if (pSPARC->wannierMMNAMNFlag) {
        // call wannier_setup_ to generate wannier inputs
    }

    t2 = MPI_Wtime();
    if (!rank)
        printf("Finish generating wannier inputs ... \n");
}

void Calculate_MMN(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1, t2;
    if (!rank)
        printf("\nStart calculating MMN ... \n");
    t1 = MPI_Wtime();
    // calculate WANNIER MMN MATRIX

    t2 = MPI_Wtime();
    if (!rank)
        printf("Finish calculating MMN ... \n");
#ifdef DEBUG
    if (!rank)
        printf("\nTime for calculating MMN: %.3f ms\n", (t2 - t1) * 1e3);
#endif
}

void Calculate_AMN(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1, t2;
    if (!rank)
        printf("\nStart calculating AMN ... \n");
    t1 = MPI_Wtime();
    // calculate WANNIER AMN MATRIX

    t2 = MPI_Wtime();
    if (!rank)
        printf("Finish calculating AMN ... \n");
#ifdef DEBUG
    if (!rank)
        printf("\nTime for calculating AMN: %.3f ms\n", (t2 - t1) * 1e3);
#endif
}
