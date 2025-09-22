/**
 * @file    mlwf.c
 * @brief   This file contains functions to generate wannier inputs.
 *
 * @author  Zehan Liu <2311531808@qq.com>
 * Copyright (c) 2025
 */

#include "mlwf.h"
#include "initialization.h"

#include <ctype.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void Generate_Wannier_Inputs(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank)
        printf("\nStart generating wannier inputs ... \n");
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    char *seed_name = pSPARC->filename;
    int mp_grid[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int num_kpts = pSPARC->Nkpts;
    // int num_kpts = pSPARC->Nkpts_sym;

    double real_lattice[3 * 3];
    double recip_lattice[3 * 3];

    double kpt_latt[3 * num_kpts];
    // double *kpt_latt = malloc(3 * sizeof(double) * num_kpts);

    int num_bands_tot = pSPARC->Nstates;
    int num_atoms = pSPARC->n_atom;
    char atom_symbols[num_atoms][3];

    //= pSPARC->atomType; // need to be filled
    double atoms_cart[3 * num_atoms];
    // double *atoms_cart = malloc(3 * sizeof(double) * pSPARC->n_atom);
    int gamma_only = (pSPARC->isGammaPoint == 1) ? 1 : 0;
    int spinors = (pSPARC->spin_typ == 0) ? 0 : 1;

    int nntot = 0;
    int nnlist[num_kpts * num_bands_tot * 3];     // int[num_kpts][num_nnmax]
    int nncell[3 * num_kpts * num_bands_tot * 3]; // int[3][num_kpts][num_nnmax]
    int num_bands = 0;
    int num_wann = pSPARC->wannier_num_wann;
    double proj_site[3 * num_bands_tot];
    int proj_l[num_bands_tot];
    int proj_m[num_bands_tot];
    int proj_radial[num_bands_tot];
    double proj_z[3 * num_bands_tot];
    double proj_x[3 * num_bands_tot];
    double proj_zona[num_bands_tot];
    int exclude_bands[num_bands_tot];
    int proj_s[num_bands_tot];              // optional
    double proj_s_qaxis[3 * num_bands_tot]; // optional
    int seed_name_len = strlen(seed_name);
    // int atom_symbols_len = strlen(atom_symbols);
    int atom_symbols_len = 3;

    // Convert units from Bohr to Angstrom

    if (rank == 0) {

#ifdef DEBUG
        printf("Start convert unit \n");
#endif

        for (int i = 0; i < pSPARC->Ntypes; i++) {
            int index = 0;
            if (i > 0) {
                for (int k = 0; k < i; k++)
                    index += pSPARC->nAtomv[k];
            }
            for (int j = 0; j < pSPARC->nAtomv[i]; j++) {
                memcpy(atom_symbols[j + index],
                       pSPARC->atomType + i * L_ATMTYPE, 2);
                // atom_symbols[j + index][0] = pSPARC->atomType[i * L_ATMTYPE];
                // atom_symbols[j + index][1] =
                // pSPARC->atomType[i * L_ATMTYPE + 1];
            }
        }

        Get_All_Cart_Coord(pSPARC);

        memcpy(recip_lattice, pSPARC->reciLattice, 9 * sizeof(double));

        double a1_x;
        double a1_y;
        double a1_z;
        double a2_x;
        double a2_y;
        double a2_z;
        double a3_x;
        double a3_y;
        double a3_z;

        double Lx = pSPARC->latvec_scale_x;
        double Ly = pSPARC->latvec_scale_y;
        double Lz = pSPARC->latvec_scale_z;

        if (pSPARC->Flag_latvec_scale) {
            double Lx = pSPARC->latvec_scale_x * CONST_BOHR;
            double Ly = pSPARC->latvec_scale_y * CONST_BOHR;
            double Lz = pSPARC->latvec_scale_z * CONST_BOHR;
            a1_x = pSPARC->LatVec[0] * Lx;
            a1_y = pSPARC->LatVec[1] * Lx;
            a1_z = pSPARC->LatVec[2] * Lx;
            a2_x = pSPARC->LatVec[3] * Ly;
            a2_y = pSPARC->LatVec[4] * Ly;
            a2_z = pSPARC->LatVec[5] * Ly;
            a3_x = pSPARC->LatVec[6] * Lz;
            a3_y = pSPARC->LatVec[7] * Lz;
            a3_z = pSPARC->LatVec[8] * Lz;
        }

        real_lattice[0] = a1_x;
        real_lattice[3] = a1_y;
        real_lattice[6] = a1_z;

        real_lattice[1] = a2_x;
        real_lattice[4] = a2_y;
        real_lattice[7] = a2_z;

        real_lattice[2] = a3_x;
        real_lattice[5] = a3_y;
        real_lattice[8] = a3_z;

        double volume = 0.0;
        double crossx, crossy, crossz;

        Cross_Product(&crossx, &crossy, &crossz, a2_x, a2_y, a2_z, a3_x, a3_y,
                      a3_z);

        volume = crossx * a1_x + crossy * a1_y + crossz * a1_z;
        // Calculate b1
        Cross_Product(&crossx, &crossy, &crossz, a2_x, a2_y, a2_z, a3_x, a3_y,
                      a3_z);
        double b1_x = 2.0 * M_PI * crossx / (volume);
        double b1_y = 2.0 * M_PI * crossy / (volume);
        double b1_z = 2.0 * M_PI * crossz / (volume);

        // Calculate b2
        Cross_Product(&crossx, &crossy, &crossz, a3_x, a3_y, a3_z, a1_x, a1_y,
                      a1_z);
        double b2_x = 2.0 * M_PI * crossx / (volume);
        double b2_y = 2.0 * M_PI * crossy / (volume);
        double b2_z = 2.0 * M_PI * crossz / (volume);

        // Calculate b3
        Cross_Product(&crossx, &crossy, &crossz, a1_x, a1_y, a1_z, a2_x, a2_y,
                      a2_z);
        double b3_x = 2.0 * M_PI * crossx / (volume);
        double b3_y = 2.0 * M_PI * crossy / (volume);
        double b3_z = 2.0 * M_PI * crossz / (volume);

        recip_lattice[0] = b1_x;
        recip_lattice[3] = b1_y;
        recip_lattice[6] = b1_z;

        recip_lattice[1] = b2_x;
        recip_lattice[4] = b2_y;
        recip_lattice[7] = b2_z;

        recip_lattice[2] = b3_x;
        recip_lattice[5] = b3_y;
        recip_lattice[8] = b3_z;

        memcpy(atoms_cart, pSPARC->atom_pos,
               3 * sizeof(double) * pSPARC->n_atom);

        for (int i = 0; i < 3 * pSPARC->n_atom; i++) {
            atoms_cart[i] *= CONST_BOHR;
        }

        for (int i = 0; i < num_kpts; i++) {
            kpt_latt[i * 3] = pSPARC->k1_fc[i];
            kpt_latt[i * 3 + 1] = pSPARC->k2_fc[i];
            kpt_latt[i * 3 + 2] = pSPARC->k3_fc[i];
        }

#ifdef DEBUG
        printf("atoms_cart: \n");
        for (int i = 0; i < 3 * num_atoms; i++) {
            printf("%4.8f  ", atoms_cart[i]);
            if ((i + 1) % 3 == 0)
                printf("\n");
        }

        printf("real_lattice: \n");
        for (int i = 0; i < 9; i++) {
            printf("%4.8f  ", real_lattice[i]);
            if ((i + 1) % 3 == 0)
                printf("\n");
        }
        printf("recip_lattice: \n");
        for (int i = 0; i < 9; i++) {
            printf("%4.8f  ", recip_lattice[i]);
            if ((i + 1) % 3 == 0)
                printf("\n");
        }

        printf("kpt_latt: \n");
        for (int i = 0; i < num_kpts; i++) {
            printf("kpt %d: %4.8f  %4.8f  %4.8f \n", i, kpt_latt[i * 3],
                   kpt_latt[i * 3 + 1], kpt_latt[i * 3 + 2]);
        }

        printf("num_kpts: %d \n", num_kpts);
        printf("num_bands_tot: %d \n", num_bands_tot);
        printf("num_atoms: %d \n", num_atoms);

        printf("atom_symbols: ");
        for (int i = 0; i < num_atoms; i++)
            printf("%s", atom_symbols[i]);
        printf("\n");

        printf("gamma_only: %d \n", gamma_only);
        printf("spinors: %d \n", spinors);

        if (rank == 0)
            printf("End convert unit \n");
#endif

        wannier_setup_(seed_name, mp_grid, &num_kpts, real_lattice,
                       recip_lattice, kpt_latt, &num_bands_tot, &num_atoms,
                       atom_symbols, atoms_cart, &gamma_only, &spinors, &nntot,
                       nnlist, nncell, &num_bands, &num_wann, proj_site, proj_l,
                       proj_m, proj_radial, proj_z, proj_x, proj_zona,
                       exclude_bands,
                       proj_s,       // optional
                       proj_s_qaxis, // optional
                       seed_name_len, atom_symbols_len);
        // print wannier outputs
        printf("nntot: %d \n", nntot);
        printf("nnlist: \n");
        for (int i = 0; i < num_kpts; i++) {
            for (int j = 0; j < nntot; j++) {
                printf("%d  ", nnlist[i * nntot + j]);
            }
            printf("\n");
        }
        printf("nncell: \n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < num_kpts; j++) {
                for (int k = 0; k < nntot; k++) {
                    printf("%d  ", nncell[i * num_kpts * nntot + j * nntot + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("num_bands: %d \n", num_bands);
        printf("num_wann: %d \n", num_wann);
        printf("proj_site: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%4.8f  %4.8f  %4.8f \n", proj_site[i * 3],
                   proj_site[i * 3 + 1], proj_site[i * 3 + 2]);
        }
        printf("proj_l: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%d  ", proj_l[i]);
        }
        printf("\n");
        printf("proj_m: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%d  ", proj_m[i]);
        }
        printf("\n");
        printf("proj_radial: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%d  ", proj_radial[i]);
        }
        printf("\n");
        printf("proj_z: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%4.8f  ", proj_z[i]);
        }
        printf("\n");
        printf("proj_x: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%4.8f  ", proj_x[i]);
        }
        printf("\n");
        printf("proj_zona: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%4.8f  ", proj_zona[i]);
        }
        printf("\n");
        printf("exclude_bands: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%d  ", exclude_bands[i]);
        }
        printf("\n");
        printf("proj_s: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%d  ", proj_s[i]);
        }
        printf("\n");
        printf("proj_s_qaxis: \n");
        for (int i = 0; i < num_bands; i++) {
            printf("%4.8f  %4.8f  %4.8f \n", proj_s_qaxis[i * 3],
                   proj_s_qaxis[i * 3 + 1], proj_s_qaxis[i * 3 + 2]);
        }

        // generate wannier inputs
        Calculate_MMN(pSPARC);

        Calculate_AMN(pSPARC);

        if (pSPARC->wannierMMNAMNFlag) {
            // call wannier_setup_ to generate wannier inputs
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (!rank)
        printf("\nTime for generate wannier inputs: %.3f ms\n",
               (t2 - t1) * 1e3);
#endif
    if (!rank)
        printf("Finish generating wannier inputs ... \n");
}

void Calculate_MMN(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank)
        printf("\nStart calculating MMN ... \n");
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    // calculate WANNIER MMN MATRIX

    if (!rank)
        printf("Finish calculating MMN ... \n");
#ifdef DEGUB
    t2 = MPI_Wtime();
    if (!rank)
        printf("\nTime for calculating MMN: %.3f ms\n", (t2 - t1) * 1e3);
#endif
}

void Calculate_AMN(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank)
        printf("\nStart calculating AMN ... \n");
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    // calculate WANNIER AMN MATRIX

    if (!rank)
        printf("Finish calculating AMN ... \n");
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (!rank)
        printf("\nTime for calculating AMN: %.3f ms\n", (t2 - t1) * 1e3);
#endif
}

void Get_All_Cart_Coord(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank)
        printf("\nStart getting all cartesian coordinates ... \n");

    for (int i = 0; i < pSPARC->n_atom; i++) {
        if (*(pSPARC->IsFrac + i) == 1) {
            nonCart2Cart_coord(pSPARC, pSPARC->atom_pos + 3 * i,
                               pSPARC->atom_pos + 3 * i + 1,
                               pSPARC->atom_pos + 3 * i + 2);
            *(pSPARC->IsFrac + i) = 0;
        }
    }

    if (!rank)
        printf("\nEnd getting all cartesian coordinates ... \n");
}
