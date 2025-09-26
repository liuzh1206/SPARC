/**
 * @file    mlwf.c
 * @brief   This file contains functions to generate wannier inputs.
 *
 * @author  Zehan Liu <2311531808@qq.com>
 * Copyright (c) 2025
 */

#include "mlwf.h"
#include "initialization.h"

#include <complex.h>
#include <ctype.h>
#include <math.h>
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
    // int num_kpts = pSPARC->Nkpts;
    int num_kpts = pSPARC->Nkpts_sym;

    double real_lattice[3 * 3];
    double recip_lattice[3 * 3];

    double kpt_latt[3 * num_kpts];

    int num_bands_tot = pSPARC->Nstates;
    int num_atoms = pSPARC->n_atom;
    char atom_symbols[num_atoms][3];

    //= pSPARC->atomType; // need to be filled
    double atoms_cart[3 * num_atoms];
    // double *atoms_cart = malloc(3 * sizeof(double) * pSPARC->n_atom);
    int gamma_only = (pSPARC->isGammaPoint == 1) ? 1 : 0;
    int spinors = (pSPARC->spin_typ == 0) ? 0 : 1;

    int nntot = 0;
    int nnlist[num_kpts * num_bands_tot];
    int nncell[3 * num_kpts * num_bands_tot];
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

    char win_filename[L_STRING];
    snprintf(win_filename, L_STRING, "%s.win", pSPARC->filename);

    // Convert units from Bohr to Angstrom

#ifdef DEBUG
    if (!rank)
        printf("Start convert unit \n");
#endif

    for (int i = 0; i < num_atoms; i++) {
        atom_symbols[i][0] = '\0';
        atom_symbols[i][1] = '\0';
        atom_symbols[i][2] = '\0';
    }
    for (int i = 0; i < pSPARC->Ntypes; i++) {
        int index = 0;
        if (i > 0) {
            for (int k = 0; k < i; k++)
                index += pSPARC->nAtomv[k];
        }
        for (int j = 0; j < pSPARC->nAtomv[i]; j++) {
            memcpy(atom_symbols[j + index], pSPARC->atomType + i * L_ATMTYPE,
                   2);
        }
    }

    double a1_x;
    double a1_y;
    double a1_z;
    double a2_x;
    double a2_y;
    double a2_z;
    double a3_x;
    double a3_y;
    double a3_z;

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

    if (!rank) {
        for (int i = 0; i < pSPARC->n_atom; i++) {
            printf("atom %d: %4.8f  %4.8f  %4.8f, IsFrac = %d\n", i,
                   pSPARC->atom_pos[3 * i], pSPARC->atom_pos[3 * i + 1],
                   pSPARC->atom_pos[3 * i + 2], pSPARC->IsFrac[i]);
        }
    }

    for (int i = 0; i < num_atoms; i++) {
        if (pSPARC->IsFrac[i] == 1) {
            atoms_cart[3 * i] = pSPARC->atom_pos[3 * i] * a1_x +
                                pSPARC->atom_pos[3 * i + 1] * a2_x +
                                pSPARC->atom_pos[3 * i + 2] * a3_x;
            atoms_cart[3 * i + 1] = pSPARC->atom_pos[3 * i] * a1_y +
                                    pSPARC->atom_pos[3 * i + 1] * a2_y +
                                    pSPARC->atom_pos[3 * i + 2] * a3_y;
            atoms_cart[3 * i + 2] = pSPARC->atom_pos[3 * i] * a1_z +
                                    pSPARC->atom_pos[3 * i + 1] * a2_z +
                                    pSPARC->atom_pos[3 * i + 2] * a3_z;
        } else {
            atoms_cart[3 * i] = pSPARC->atom_pos[3 * i] * CONST_BOHR;
            atoms_cart[3 * i + 1] = pSPARC->atom_pos[3 * i + 1] * CONST_BOHR;
            atoms_cart[3 * i + 2] = pSPARC->atom_pos[3 * i + 2] * CONST_BOHR;
        }
    }

    for (int i = 0; i < num_kpts; i++) {
        kpt_latt[i * 3] = pSPARC->k1_fc[i];
        kpt_latt[i * 3 + 1] = pSPARC->k2_fc[i];
        kpt_latt[i * 3 + 2] = pSPARC->k3_fc[i];
    }

#ifdef DEBUG
    if (!rank) {
        t2 = MPI_Wtime();
        printf("\nTime for convert unit: %.3f ms\n", (t2 - t1) * 1e3);

        printf("atoms_cart: \n");
        for (int i = 0; i < 3 * num_atoms; i++) {
            printf("%4.8f  ", atoms_cart[i]);
            if ((i + 1) % 3 == 0)
                printf("\n");
        }

        printf("real_lattice: \n");
        for (int i = 0; i < 3; i++) {
            printf("%4.8f  %4.8f  %4.8f \n", real_lattice[i],
                   real_lattice[i + 3], real_lattice[i + 6]);
        }
        printf("recip_lattice: \n");
        for (int i = 0; i < 3; i++) {
            printf("%4.8f  %4.8f  %4.8f \n", recip_lattice[i],
                   recip_lattice[i + 3], recip_lattice[i + 6]);
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

        printf("End convert unit\n");
    }
#endif
    // generate wannier inputs
    if (!rank) {

        FILE *fp_win = fopen(win_filename, "w");
        fprintf(fp_win, "# Generated by SPARC\n");
        fprintf(fp_win, "num_bands = %d\n", num_bands_tot);
        fprintf(fp_win, "num_wann = %d\n", num_wann);
        fprintf(fp_win, "begin unit_cell_cart\n");
        for (int i = 0; i < 3; i++) {
            fprintf(fp_win, "  %15.7f  %15.7f  %15.7f\n", real_lattice[i],
                    real_lattice[i + 3], real_lattice[i + 6]);
        }
        fprintf(fp_win, "end unit_cell_cart\n");
        fprintf(fp_win, "begin atoms_cart\n");
        for (int i = 0; i < num_atoms; i++) {
            fprintf(fp_win, "  %s  %15.7f  %15.7f  %15.7f\n", atom_symbols[i],
                    atoms_cart[i * 3], atoms_cart[i * 3 + 1],
                    atoms_cart[i * 3 + 2]);
        }
        fprintf(fp_win, "end atoms_cart\n");
        fprintf(fp_win, "begin kpoints\n");
        for (int i = 0; i < num_kpts; i++) {
            fprintf(fp_win, "  %15.7f  %15.7f  %15.7f\n", kpt_latt[i * 3],
                    kpt_latt[i * 3 + 1], kpt_latt[i * 3 + 2]);
        }
        fprintf(fp_win, "end kpoints\n");
        fclose(fp_win);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    wannier_setup_(seed_name, mp_grid, &num_kpts, real_lattice, recip_lattice,
                   kpt_latt, &num_bands_tot, &num_atoms, atom_symbols,
                   atoms_cart, &gamma_only, &spinors, &nntot, nnlist, nncell,
                   &num_bands, &num_wann, proj_site, proj_l, proj_m,
                   proj_radial, proj_z, proj_x, proj_zona, exclude_bands,
                   proj_s,       // optional
                   proj_s_qaxis, // optional
                   seed_name_len, atom_symbols_len);
    if (!rank) {
        FILE *fp_win = fopen(win_filename, "a");
        /* fprintf(fp_win, "num_bands = %d\n", num_bands); */
        fclose(fp_win);
    }
#ifdef DEBUG
    if (!rank) {
        t1 = t2;
        t2 = MPI_Wtime();
        printf("\nTime for wannier_setup_: %.3f ms\n", (t2 - t1) * 1e3);
        // print wannier outputs
        printf("nntot: %5d\n", nntot);
        printf("nnlist:\n");
        for (int i = 0; i < num_kpts; i++) {
            for (int j = 0; j < nntot; j++) {
                printf("%5d", nnlist[i * nntot + j]);
            }
            printf("\n");
        }
        printf("nncell:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < num_kpts; j++) {
                for (int k = 0; k < nntot; k++) {
                    printf("%5d", nncell[i * num_kpts * nntot + j * nntot + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("num_bands: %d\n", num_bands);
        printf("num_wann: %d\n", num_wann);
        printf("proj_site:\n");
        for (int i = 0; i < num_bands; i++) {
            printf(" %4.8f  %4.8f  %4.8f\n", proj_site[i * 3],
                   proj_site[i * 3 + 1], proj_site[i * 3 + 2]);
        }
        printf("proj_l:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %7d", proj_l[i]);
        }
        printf("\n");
        printf("proj_m:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %7d", proj_m[i]);
        }
        printf("\n");
        printf("proj_radial:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %7d", proj_radial[i]);
        }
        printf("\n");
        printf("proj_z:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %4.8f", proj_z[i]);
        }
        printf("\n");
        printf("proj_x:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %4.8f", proj_x[i]);
        }
        printf("\n");
        printf("proj_zona:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %4.8f", proj_zona[i]);
        }
        printf("\n");
        printf("exclude_bands:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %7d", exclude_bands[i]);
        }
        printf("\n");
        printf("proj_s:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %7d", proj_s[i]);
        }
        printf("\n");
        printf("proj_s_qaxis:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("%4.8f  %4.8f  %4.8f\n", proj_s_qaxis[i * 3],
                   proj_s_qaxis[i * 3 + 1], proj_s_qaxis[i * 3 + 2]);
        }
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    double complex MMN_Matrix[num_kpts][nntot][num_bands][num_bands];

    // generate wannier inputs
    Calculate_MMN(pSPARC, num_kpts, nntot, nnlist, nncell, num_bands,
                  MMN_Matrix);

    Calculate_AMN(pSPARC);

    if (!rank) {
        if (pSPARC->wannierMMNAMNFlag) {
            char mmn_filename[L_STRING];
            snprintf(mmn_filename, L_STRING, "%s.mmn", pSPARC->filename);
            FILE *fp_mmn = fopen(mmn_filename, "w");
            fprintf(fp_mmn, "Generated by SPARC\n");
            fprintf(fp_mmn, "%12d %12d %12d\n", num_bands, num_kpts, nntot);

            if (pSPARC->spin_typ == 0) {
                for (int kpt = 0; kpt < num_kpts; kpt++) {
                    for (int nn = 0; nn < nntot; nn++) {
                        int image_index = nnlist[kpt * nntot + nn];
                        int n1 =
                            nncell[0 * num_kpts * nntot + kpt * nntot + nn];
                        int n2 =
                            nncell[1 * num_kpts * nntot + kpt * nntot + nn];
                        int n3 =
                            nncell[2 * num_kpts * nntot + kpt * nntot + nn];
                        fprintf(fp_mmn, "%5d %5d %5d %5d %5d\n", kpt + 1,
                                image_index, n1, n2, n3);
                        for (int m = 0; m < num_bands; m++) {
                            for (int n = 0; n < num_bands; n++) {
                                double complex mmn_element =
                                    MMN_Matrix[kpt][nn][m][n];

                                fprintf(fp_mmn, "%18.12f %18.12f\n",
                                        creal(mmn_element), cimag(mmn_element));
                            }
                        }
                    }
                }

                fclose(fp_mmn);
            }
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (!rank)
        printf("\nTime for generate wannier inputs: %.3f ms\n",
               (t2 - t1) * 1e3);
#endif
    if (!rank)
        printf("Finish generating wannier inputs ...\n");
}

void Calculate_MMN(SPARC_OBJ *pSPARC, int num_kpts, int nntot, int *nnlist,
                   int *nncell, int num_bands,
                   double complex MMN_Matrix[][nntot][num_bands][num_bands]) {
    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (!rank)
        printf("\nStart calculating MMN ...\n");
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    // calculate WANNIER MMN MATRIX

    int spin = 1;
    if (pSPARC->spin_typ != 0)
        spin = 2;

    int sendcounts = pSPARC->Nd_d_dmcomm * pSPARC->Nspin_spincomm *
                     pSPARC->Nband_bandcomm * pSPARC->Nkpts_kptcomm;
    int total_size = pSPARC->Nkpts_sym * spin * pSPARC->Nstates * pSPARC->Nd;

    int *Nd_d_dmcomm;
    int *Nspinor_spincomm;
    int *Nband_bandcomm;
    int *Nkpts_kptcomm;

    int *dispos;
    int *recvcounts;
    double complex *Xorb_kpt_all;

    if (!rank) {
        Nd_d_dmcomm = (int *)malloc(size * sizeof(int));
        Nspinor_spincomm = (int *)malloc(size * sizeof(int));
        Nband_bandcomm = (int *)malloc(size * sizeof(int));
        Nkpts_kptcomm = (int *)malloc(size * sizeof(int));
        dispos = (int *)malloc(size * sizeof(int));
        recvcounts = (int *)malloc(size * sizeof(int));
        Xorb_kpt_all =
            (double complex *)malloc(total_size * sizeof(double complex));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(&(pSPARC->Nd_d_dmcomm), 1, MPI_INT, Nd_d_dmcomm, 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    MPI_Gather(&(pSPARC->Nspinor_spincomm), 1, MPI_INT, Nspinor_spincomm, 1,
               MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gather(&(pSPARC->Nband_bandcomm), 1, MPI_INT, Nband_bandcomm, 1,
               MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gather(&(pSPARC->Nkpts_kptcomm), 1, MPI_INT, Nkpts_kptcomm, 1, MPI_INT,
               0, MPI_COMM_WORLD);
#ifdef DEBUG
    if (!rank) {
        for (int i = 0; i < size; i++) {
            printf("rank %d: Nd_d_dmcomm = %d, Nspinor_spincomm = %d, "
                   "Nband_bandcomm = %d, Nkpts_kptcomm = %d\n",
                   i, Nd_d_dmcomm[i], Nspinor_spincomm[i], Nband_bandcomm[i],
                   Nkpts_kptcomm[i]);
        }
    }
#endif
    if (!rank) {
        dispos[0] = 0;
        recvcounts[0] = Nd_d_dmcomm[0] * Nspinor_spincomm[0] *
                        Nband_bandcomm[0] * Nkpts_kptcomm[0];
        for (int i = 1; i < size; i++) {
            dispos[i] = dispos[i - 1] +
                        Nd_d_dmcomm[i - 1] * Nspinor_spincomm[i - 1] *
                            Nband_bandcomm[i - 1] * Nkpts_kptcomm[i - 1];
            recvcounts[i] = Nd_d_dmcomm[i] * Nspinor_spincomm[i] *
                            Nband_bandcomm[i] * Nkpts_kptcomm[i];
        }

        for (int i = 0; i < size; i++) {
            printf("rank = %d sidpos= %d, recvcounts = %d\n", i, dispos[i],
                   recvcounts[i]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(pSPARC->Xorb_kpt, sendcounts, MPI_DOUBLE_COMPLEX, Xorb_kpt_all,
                recvcounts, dispos, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (!rank) {

        double a1_x;
        double a1_y;
        double a1_z;
        double a2_x;
        double a2_y;
        double a2_z;
        double a3_x;
        double a3_y;
        double a3_z;

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

        if (pSPARC->spin_typ == 0) {
            for (int kpt = 0; kpt < num_kpts; kpt++) {
                for (int nn = 0; nn < nntot; nn++) {

                    int image_index = nnlist[kpt * nntot + nn] - 1;

                    int nx = nncell[0 * num_kpts * nntot + kpt * nntot + nn];
                    int ny = nncell[1 * num_kpts * nntot + kpt * nntot + nn];
                    int nz = nncell[2 * num_kpts * nntot + kpt * nntot + nn];

                    double bx = pSPARC->k1[image_index] - pSPARC->k1[kpt];
                    double by = pSPARC->k2[image_index] - pSPARC->k2[kpt];
                    double bz = pSPARC->k3[image_index] - pSPARC->k3[kpt];

                    double phi = bx * (nx * a1_x + ny * a2_x + nz * a3_x) +
                                 by * (nx * a1_y + ny * a2_y + nz * a3_y) +
                                 bz * (nx * a1_z + ny * a2_z + nz * a3_z);

                    // double complex psi = cos(phi) + I * sin(phi);

                    for (int m = 0; m < num_bands; m++) {
                        for (int n = 0; n < num_bands; n++) {
                            // MMN matrix element between band m and n at kpt
                            // and image_index
                            double complex mmn_element = 0.0 + 0.0 * I;
                            for (int i = 0; i < pSPARC->Nd; i++) {
                                // pSPARC->Xorb_kpt[];
                                mmn_element +=
                                    /* conj(Xorb_kpt_all[kpt * pSPARC->Nstates *
                                     */
                                    /*                       pSPARC->Nd + */
                                    /*                   m * pSPARC->Nd + i]) *
                                     */
                                    /* (sin(phi) + I * cos(phi)) * */
                                    /* Xorb_kpt_all[kpt * pSPARC->Nd * */
                                    /*                  pSPARC->Nstates + */
                                    /*              n * pSPARC->Nd + i]; */
                                    conj(Xorb_kpt_all[kpt * pSPARC->Nd *
                                                          pSPARC->Nstates +
                                                      m * pSPARC->Nd + i]) *
                                    (sin(phi) + I * cos(phi)) *
                                    Xorb_kpt_all[kpt * pSPARC->Nd *
                                                     pSPARC->Nstates +
                                                 n * pSPARC->Nd + i];
                            }
                            // mmn_element /= pSPARC->Nd; // Normalize
                            MMN_Matrix[kpt][nn][m][n] = mmn_element;
                        }
                    }
                }
            }
        }
    }

    if (!rank)
        printf("Finish calculating MMN ...\n");
#ifdef DEBUG
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

    double complex x = 1.0 + 2.0 * I;
    double complex y = 2.0 + 1.0 * I;
    if (!rank) {
        printf("complex test: %4.8f + %4.8fi \n", creal(x), cimag(x));
        printf("conf %f, %f\n", creal(conj(x) * (y)), cimag(conj(y) * (x)));
        printf("Finish calculating AMN ... \n");
    }
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (!rank)
        printf("\nTime for calculating AMN: %.3f ms\n", (t2 - t1) * 1e3);
#endif
}
