/**
 * @file    mlwf.c
 * @brief   This file contains functions to generate wannier inputs.
 *
 * @author  Zehan Liu <2311531808@qq.com>
 * Copyright (c) 2025
 */

#include "mlwf.h"
#include "initialization.h"
#include "parallelization.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))

void Generate_Wannier_Inputs(SPARC_OBJ *pSPARC) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank)
        printf("\nStart generating wannier inputs ... \n");

    Write_EIG(pSPARC);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    char *seed_name = pSPARC->filename;
    // int mp_grid[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int mp_grid[3] = {pSPARC->Kx, pSPARC->Ky, pSPARC->Kz};
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
    // int spinors = (pSPARC->spin_typ == 0) ? 0 : 1;
    int spinors = (pSPARC->Nspinor_eig == 1) ? 0 : 1;

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
        atom_symbols[i][2] = ' ';
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

    double a1[3];
    double a2[3];
    double a3[3];

    double Lx = pSPARC->latvec_scale_x * CONST_BOHR;
    double Ly = pSPARC->latvec_scale_y * CONST_BOHR;
    double Lz = pSPARC->latvec_scale_z * CONST_BOHR;

    a1[0] = pSPARC->LatVec[0] * Lx;
    a1[1] = pSPARC->LatVec[1] * Lx;
    a1[2] = pSPARC->LatVec[2] * Lx;
    a2[0] = pSPARC->LatVec[3] * Ly;
    a2[1] = pSPARC->LatVec[4] * Ly;
    a2[2] = pSPARC->LatVec[5] * Ly;
    a3[0] = pSPARC->LatVec[6] * Lz;
    a3[1] = pSPARC->LatVec[7] * Lz;
    a3[2] = pSPARC->LatVec[8] * Lz;

    real_lattice[0] = a1[0];
    real_lattice[3] = a1[1];
    real_lattice[6] = a1[2];

    real_lattice[1] = a2[0];
    real_lattice[4] = a2[1];
    real_lattice[7] = a2[2];

    real_lattice[2] = a3[0];
    real_lattice[5] = a3[1];
    real_lattice[8] = a3[2];

    double b1[3];
    double b2[3];
    double b3[3];
    double volume = 0.0;
    double crossx, crossy, crossz;

    Cross_Product(&crossx, &crossy, &crossz, a2[0], a2[1], a2[2], a3[0], a3[1],
                  a3[2]);

    volume = crossx * a1[0] + crossy * a1[1] + crossz * a1[2];
    // Calculate b1
    Cross_Product(&crossx, &crossy, &crossz, a2[0], a2[1], a2[2], a3[0], a3[1],
                  a3[2]);
    b1[0] = 2.0 * M_PI * crossx / (volume);
    b1[1] = 2.0 * M_PI * crossy / (volume);
    b1[2] = 2.0 * M_PI * crossz / (volume);

    // Calculate b2
    Cross_Product(&crossx, &crossy, &crossz, a3[0], a3[1], a3[2], a1[0], a1[1],
                  a1[2]);
    b2[0] = 2.0 * M_PI * crossx / (volume);
    b2[1] = 2.0 * M_PI * crossy / (volume);
    b2[2] = 2.0 * M_PI * crossz / (volume);

    // Calculate b3
    Cross_Product(&crossx, &crossy, &crossz, a1[0], a1[1], a1[2], a2[0], a2[1],
                  a2[2]);
    b3[0] = 2.0 * M_PI * crossx / (volume);
    b3[1] = 2.0 * M_PI * crossy / (volume);
    b3[2] = 2.0 * M_PI * crossz / (volume);

    recip_lattice[0] = b1[0];
    recip_lattice[3] = b1[1];
    recip_lattice[6] = b1[2];

    recip_lattice[1] = b2[0];
    recip_lattice[4] = b2[1];
    recip_lattice[7] = b2[2];

    recip_lattice[2] = b3[0];
    recip_lattice[5] = b3[1];
    recip_lattice[8] = b3[2];

    if (!rank) {
        for (int i = 0; i < pSPARC->n_atom; i++) {
            printf("atom %d: %4.8f  %4.8f  %4.8f, IsFrac = %d\n", i,
                   pSPARC->atom_pos[3 * i], pSPARC->atom_pos[3 * i + 1],
                   pSPARC->atom_pos[3 * i + 2], pSPARC->IsFrac[i]);
        }
    }

    for (int i = 0; i < num_atoms; i++) {
        if (pSPARC->IsFrac[i] == 1) {
            atoms_cart[3 * i] = pSPARC->atom_pos[3 * i] * a1[0] +
                                pSPARC->atom_pos[3 * i + 1] * a2[0] +
                                pSPARC->atom_pos[3 * i + 2] * a3[0];
            atoms_cart[3 * i + 1] = pSPARC->atom_pos[3 * i] * a1[1] +
                                    pSPARC->atom_pos[3 * i + 1] * a2[1] +
                                    pSPARC->atom_pos[3 * i + 2] * a3[1];
            atoms_cart[3 * i + 2] = pSPARC->atom_pos[3 * i] * a1[2] +
                                    pSPARC->atom_pos[3 * i + 1] * a2[2] +
                                    pSPARC->atom_pos[3 * i + 2] * a3[2];
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

        fprintf(fp_win, "# SPARC wannier90 input file\n");
        fprintf(fp_win, "%s", pSPARC->wannier_win);

        fprintf(fp_win, "# Generated by SPARC\n");
        fprintf(fp_win, "num_bands = %d\n", num_bands_tot);
        fprintf(fp_win, "num_wann = %d\n", num_wann);
        fprintf(fp_win, "mp_grid = %d %d %d\n", mp_grid[0], mp_grid[1],
                mp_grid[2]);
        fprintf(fp_win, "begin unit_cell_cart\n");
        for (int i = 0; i < 3; i++) {
            fprintf(fp_win, "  %15.7f  %15.7f  %15.7f\n", real_lattice[i],
                    real_lattice[i + 3], real_lattice[i + 6]);
        }
        fprintf(fp_win, "end unit_cell_cart\n");
        fprintf(fp_win, "begin atoms_cart\n");
        for (int i = 0; i < num_atoms; i++) {
            fprintf(fp_win, "%s  %15.7f  %15.7f  %15.7f\n", atom_symbols[i],
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
            printf("  %13.8f%13.8f%13.8f\n", proj_z[3 * i], proj_z[3 * i + 1],
                   proj_z[3 * i + 2]);
        }
        printf("\n");
        printf("proj_x:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %13.8f%13.8f%13.8f\n", proj_x[3 * i], proj_x[3 * i + 1],
                   proj_x[3 * i + 2]);
        }
        printf("\n");
        printf("proj_zona:\n");
        for (int i = 0; i < num_bands; i++) {
            printf("  %13.8f", proj_zona[i]);
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
            printf("%13.8f  %13.8f  %13.8f\n", proj_s_qaxis[i * 3],
                   proj_s_qaxis[i * 3 + 1], proj_s_qaxis[i * 3 + 2]);
        }
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    double complex MMN_Matrix[num_kpts * pSPARC->Nspinor_eig * pSPARC->Nspin]
                             [nntot][num_bands][num_bands];

    size_t amn_size =
        pSPARC->Nspin * pSPARC->Nspinor_eig * num_kpts * num_bands * num_wann;

    double complex *AMN_Matrix =
        (double complex *)calloc(amn_size, sizeof(double complex));

    if (!AMN_Matrix) {
        fprintf(stderr, "Failed to allocate AMN_Matrix\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // generate wannier inputs
    Calculate_MMN(pSPARC, num_kpts, nntot, nnlist, nncell, num_bands,
                  MMN_Matrix);

    Calculate_AMN(pSPARC, num_bands, num_kpts, num_wann, exclude_bands,
                  proj_site, proj_l, proj_m, proj_z, proj_x, proj_zona,
                  gamma_only, spinors, AMN_Matrix);

    if (!rank) {
        if (pSPARC->wannierMMNAMNFlag) {

            // write MMN file
            char mmn_filename[L_STRING];
            char mmn_filename_up[L_STRING];
            char mmn_filename_dn[L_STRING];
            FILE *fp_mmn = NULL;
            FILE *fp_mmn_up = NULL;
            FILE *fp_mmn_dn = NULL;
            if (pSPARC->spin_typ == 0) {
                printf("Writing MMN file for non-spinor calculation ...\n");
                snprintf(mmn_filename, L_STRING, "%s.mmn", pSPARC->filename);
                fp_mmn = fopen(mmn_filename, "w");
                fprintf(fp_mmn, "Generated by SPARC\n");
                fprintf(fp_mmn, "%12d %12d %12d\n", num_bands, num_kpts, nntot);
            } else {
                printf("Writing MMN file for spinor calculation ...\n");
                snprintf(mmn_filename_up, L_STRING, "%s_up.mmn",
                         pSPARC->filename);
                snprintf(mmn_filename_dn, L_STRING, "%s_dn.mmn",
                         pSPARC->filename);
                fp_mmn_up = fopen(mmn_filename_up, "w");
                fp_mmn_dn = fopen(mmn_filename_dn, "w");
                fprintf(fp_mmn_up, "Generated by SPARC\n");
                fprintf(fp_mmn_up, "%12d %12d %12d\n", num_bands, num_kpts,
                        nntot);
                fprintf(fp_mmn_dn, "Generated by SPARC\n");
                fprintf(fp_mmn_dn, "%12d %12d %12d\n", num_bands, num_kpts,
                        nntot);
            }

            for (int kpt = 0; kpt < num_kpts; kpt++) {
                for (int nn = 0; nn < nntot; nn++) {
                    int image_index = nnlist[kpt * nntot + nn];
                    int n1 = nncell[0 * num_kpts * nntot + kpt * nntot + nn];
                    int n2 = nncell[1 * num_kpts * nntot + kpt * nntot + nn];
                    int n3 = nncell[2 * num_kpts * nntot + kpt * nntot + nn];
                    if (pSPARC->spin_typ == 0)
                        fprintf(fp_mmn, "%5d %5d %5d %5d %5d\n", kpt + 1,
                                image_index, n1, n2, n3);
                    else {
                        fprintf(fp_mmn_up, "%5d %5d %5d %5d %5d\n", kpt + 1,
                                image_index, n1, n2, n3);
                        fprintf(fp_mmn_dn, "%5d %5d %5d %5d %5d\n", kpt + 1,
                                image_index, n1, n2, n3);
                    }
                    for (int m = 0; m < num_bands; m++) {
                        for (int n = 0; n < num_bands; n++) {
                            double complex mmn_element =
                                MMN_Matrix[kpt][nn][m][n];

                            if (pSPARC->spin_typ == 0)
                                fprintf(fp_mmn, "%18.12f %18.12f\n",
                                        creal(mmn_element), cimag(mmn_element));
                            else {
                                for (int spin = 0; spin < pSPARC->Nspin;
                                     spin++) {
                                    for (int s = 0; s < pSPARC->Nspinor_eig;
                                         s++) {
                                        mmn_element =
                                            MMN_Matrix[spin * pSPARC->Nkpts *
                                                           pSPARC->Nspinor_eig +
                                                       kpt *
                                                           pSPARC->Nspinor_eig +
                                                       s][nn][m][n];
                                        if (spin == 0 && s == 0) {
                                            fprintf(fp_mmn_up,
                                                    "%18.12f %18.12f\n",
                                                    creal(mmn_element),
                                                    cimag(mmn_element));
                                        } else {
                                            fprintf(fp_mmn_dn,
                                                    "%18.12f %18.12f\n",
                                                    creal(mmn_element),
                                                    cimag(mmn_element));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (pSPARC->spin_typ == 0)
                fclose(fp_mmn);
            else {
                fclose(fp_mmn_up);
                fclose(fp_mmn_dn);
            }
            // write MMN file
            char amn_filename[L_STRING];
            char amn_filename_up[L_STRING];
            char amn_filename_dn[L_STRING];
            FILE *fp_amn = NULL;
            FILE *fp_amn_up = NULL;
            FILE *fp_amn_dn = NULL;
            if (pSPARC->spin_typ == 0) {
                printf("Writing AMN file for non-spinor calculation ...\n");
                snprintf(amn_filename, L_STRING, "%s.amn", pSPARC->filename);
                fp_amn = fopen(amn_filename, "w");
                fprintf(fp_amn, "Generated by SPARC\n");
                fprintf(fp_amn, "%12d %12d %12d\n", num_bands, num_kpts,
                        num_wann);
            } else {
                printf("Writing AMN file for spinor calculation ...\n");
                snprintf(amn_filename_up, L_STRING, "%s_up.amn",
                         pSPARC->filename);
                snprintf(amn_filename_dn, L_STRING, "%s_dn.amn",
                         pSPARC->filename);
                fp_mmn_up = fopen(amn_filename_up, "w");
                fp_mmn_dn = fopen(amn_filename_dn, "w");
                fprintf(fp_amn_up, "Generated by SPARC\n");
                fprintf(fp_amn_dn, "Generated by SPARC\n");
                fprintf(fp_amn_up, "%12d %12d %12d\n", num_bands, num_kpts,
                        num_wann);
                fprintf(fp_amn_dn, "%12d %12d %12d\n", num_bands, num_kpts,
                        num_wann);
            }

            for (int band = 0; band < num_bands; band++) {
                for (int iw = 0; iw < num_wann; iw++) {
                    for (int kpt = 0; kpt < num_kpts; kpt++) {

                        if (pSPARC->spin_typ == 0) {
                            double complex amn_element =
                                AMN_Matrix[kpt * num_bands * num_wann +
                                           band * num_wann + iw];
                            fprintf(fp_amn, "%5d %5d %5d %18.12f %18.12f\n",
                                    band, iw, kpt, creal(amn_element),
                                    cimag(amn_element));
                        } else {
                            for (int spin = 0; spin < pSPARC->Nspin; spin++) {
                                for (int s = 0; s < pSPARC->Nspinor_eig; s++) {
                                    double complex amn_element =
                                        AMN_Matrix[(spin + s) * num_kpts *
                                                       num_bands * num_wann +
                                                   kpt * num_bands * num_wann +
                                                   band * num_wann + iw];
                                    if (spin == 0 && s == 0) {
                                        fprintf(fp_amn_up,
                                                "%5d %5d %5d %18.12f %18.12f\n",
                                                band, iw, kpt,
                                                creal(amn_element),
                                                cimag(amn_element));
                                    } else {
                                        fprintf(fp_amn_dn,
                                                "%5d %5d %5d %18.12f %18.12f\n",
                                                band, iw, kpt,
                                                creal(amn_element),
                                                cimag(amn_element));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (pSPARC->spin_typ == 0)
                fclose(fp_amn);
            else {
                fclose(fp_amn_up);
                fclose(fp_amn_dn);
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

void Write_EIG(SPARC_OBJ *pSPARC) {

    int rank, rank_spincomm, rank_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

    if (!rank)
        printf("\nStart writing eigenvalues ...\n");

    // only root processes of kptcomms will enter
    if (pSPARC->kptcomm_index < 0 || rank_kptcomm != 0)
        return;

    int Nk = pSPARC->Nkpts_kptcomm;
    int Ns = pSPARC->Nstates;
    // number of kpoints assigned to each kptcomm
    int *Nk_i = (int *)malloc(pSPARC->npkpt * sizeof(int));
    double *kred_i = (double *)malloc(pSPARC->Nkpts_sym * 3 * sizeof(double));
    int *kpt_displs = (int *)malloc((pSPARC->npkpt + 1) * sizeof(int));

    char EigenFilename[L_STRING];
    char EigenFilename_up[L_STRING];
    char EigenFilename_dn[L_STRING];
    if (rank == 0) {
        snprintf(EigenFilename, L_STRING, "%s.eig", pSPARC->filename);
        snprintf(EigenFilename_up, L_STRING, "%s_up.eig", pSPARC->filename);
        snprintf(EigenFilename_dn, L_STRING, "%s_dn.eig", pSPARC->filename);
    }

    FILE *output_fp;
    FILE *output_fp_up;
    FILE *output_fp_dn;
    // first create an empty file
    if (rank == 0) {
        if (pSPARC->Nspin == 1) {
            output_fp = fopen(EigenFilename, "w");

            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n", EigenFilename);
                exit(EXIT_FAILURE);
            }
            fclose(output_fp);
        } else {
            output_fp_up = fopen(EigenFilename_up, "w");
            output_fp_dn = fopen(EigenFilename_dn, "w");

            if (output_fp_up == NULL) {
                printf("\nCannot open file \"%s\"\n", EigenFilename_up);
                exit(EXIT_FAILURE);
            }
            if (output_fp_dn == NULL) {
                printf("\nCannot open file \"%s\"\n", EigenFilename_dn);
                exit(EXIT_FAILURE);
            }
            fclose(output_fp_up);
            fclose(output_fp_dn);
        }
    }

    int sendcount, *recvcounts, *displs;
    double *recvbuf_eig;
    sendcount = 0;
    recvcounts = NULL;
    displs = NULL;
    recvbuf_eig = NULL;

    // first collect eigval over spin
    if (pSPARC->npspin > 1) {
        // set up receive buffer and receive counts in kptcomm roots with spin
        // up
        if (pSPARC->spincomm_index == 0) {
            recvbuf_eig =
                (double *)malloc(pSPARC->Nspin * Nk * Ns * sizeof(double));
            recvcounts =
                (int *)malloc(pSPARC->npspin * sizeof(int)); // npspin is 2
            displs = (int *)malloc((pSPARC->npspin + 1) * sizeof(int));
            int i;
            displs[0] = 0;
            for (i = 0; i < pSPARC->npspin; i++) {
                recvcounts[i] = pSPARC->Nspin_spincomm * Nk * Ns;
                displs[i + 1] = displs[i] + recvcounts[i];
            }
        }
        // set up send info
        sendcount = pSPARC->Nspin_spincomm * Nk * Ns;
        MPI_Gatherv(pSPARC->lambda_sorted, sendcount, MPI_DOUBLE, recvbuf_eig,
                    recvcounts, displs, MPI_DOUBLE, 0,
                    pSPARC->spin_bridge_comm);

        if (pSPARC->spincomm_index == 0) {
            free(recvcounts);
            free(displs);
        }
    } else {
        recvbuf_eig = pSPARC->lambda_sorted;
    }

    double *eig_all = NULL;
    int *displs_all;
    displs_all = (int *)malloc((pSPARC->npkpt + 1) * sizeof(int));

    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
        // set up receive buffer and receive counts in kptcomm roots with spin
        // up
        if (pSPARC->kptcomm_index == 0) {
            int i;
            eig_all = (double *)malloc(pSPARC->Nspin * pSPARC->Nkpts_sym * Ns *
                                       sizeof(double));
            recvcounts = (int *)malloc(pSPARC->npkpt * sizeof(int));
            // collect all the number of kpoints assigned to each kptcomm
            MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT, 0,
                       pSPARC->kpt_bridge_comm);
            displs_all[0] = 0;
            for (i = 0; i < pSPARC->npkpt; i++) {
                recvcounts[i] = Nk_i[i] * pSPARC->Nspin * Ns;
                displs_all[i + 1] = displs_all[i] + recvcounts[i];
            }
            // collect all the kpoints assigend to each kptcomm
            // first set up sendbuf and recvcounts
            int *kpt_recvcounts = (int *)malloc(pSPARC->npkpt * sizeof(int));
            // int *kpt_displs     = (int *)malloc((pSPARC->npkpt+1) *
            kpt_displs[0] = 0;
            for (i = 0; i < pSPARC->npkpt; i++) {
                kpt_recvcounts[i] = Nk_i[i] * 3;
                kpt_displs[i + 1] = kpt_displs[i] + kpt_recvcounts[i];
            }
            free(kpt_recvcounts);
        } else {
            // collect all the number of kpoints assigned to each kptcomm
            MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT, 0,
                       pSPARC->kpt_bridge_comm);
            // collect all the kpoints assigend to each kptcomm
            int kpt_recvcounts[1] = {0}, i;
            // collect reduced kpoints from all kptcomms
        }
        // set up send info
        sendcount = pSPARC->Nspin * Nk * Ns;
        MPI_Gatherv(recvbuf_eig, sendcount, MPI_DOUBLE, eig_all, recvcounts,
                    displs_all, MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
        if (pSPARC->kptcomm_index == 0) {
            free(recvcounts);
            // free(displs_all);
        }
    } else {
        int i;
        Nk_i[0] = Nk; // only one kptcomm
        kpt_displs[0] = 0;
        displs_all[0] = 0;
        if (pSPARC->BC != 1) {
            if (pSPARC->BandStructFlag == 1) {
                for (i = 0; i < Nk; i++) {
                    kred_i[3 * i] = pSPARC->k1_inpt_kpt[i];
                    kred_i[3 * i + 1] = pSPARC->k2_inpt_kpt[i];
                    kred_i[3 * i + 2] = pSPARC->k3_inpt_kpt[i];
                }
            } else {
                for (i = 0; i < Nk; i++) {
                    kred_i[3 * i] =
                        pSPARC->k1_loc[i] * pSPARC->range_x / (2.0 * M_PI);
                    kred_i[3 * i + 1] =
                        pSPARC->k2_loc[i] * pSPARC->range_y / (2.0 * M_PI);
                    kred_i[3 * i + 2] =
                        pSPARC->k3_loc[i] * pSPARC->range_z / (2.0 * M_PI);
                }
            }
        } else {
            kred_i[0] = kred_i[1] = kred_i[2] = 0.0;
        }
        eig_all = recvbuf_eig;
    }

    // let root process print eigvals and occs to .eigen file
    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            // write to .eig file
            int k, Kcomm_indx, i;
            if (pSPARC->Nspin == 1) {
                output_fp = fopen(EigenFilename, "a");
                if (output_fp == NULL) {
                    printf("\nCannot open file \"%s\"\n", EigenFilename);
                    exit(EXIT_FAILURE);
                }
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        int kred_index = kpt_displs[Kcomm_indx] / 3 + k + 1;
                        for (i = 0; i < pSPARC->Nstates; i++) {
                            fprintf(
                                output_fp, "%7d%7d%  25.12E\n", i + 1,
                                kred_index,
                                eig_all[displs_all[Kcomm_indx] + k * Ns + i]);
                        }
                    }
                }
                fclose(output_fp);
            } else if (pSPARC->Nspin == 2) {
                output_fp_up = fopen(EigenFilename_up, "a");
                output_fp_dn = fopen(EigenFilename_dn, "a");
                if (output_fp_up == NULL) {
                    printf("\nCannot open file \"%s\"\n", EigenFilename_up);
                    exit(EXIT_FAILURE);
                }
                if (output_fp_dn == NULL) {
                    printf("\nCannot open file \"%s\"\n", EigenFilename_dn);
                    exit(EXIT_FAILURE);
                }
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        int kred_index = kpt_displs[Kcomm_indx] / 3 + k + 1;
                        for (i = 0; i < pSPARC->Nstates; i++) {
                            fprintf(
                                output_fp_up, "%7d%7d     %25.12E \n", i + 1,
                                kred_index,
                                eig_all[displs_all[Kcomm_indx] + k * Ns + i]);
                            fprintf(output_fp_dn, "%7d%7d     %25.12E \n",
                                    i + 1, kred_index,
                                    eig_all[displs_all[Kcomm_indx] +
                                            (Nk_Kcomm_indx + k) * Ns + i]);
                        }
                    }
                }
                fclose(output_fp_up);
                fclose(output_fp_dn);
            }
        }
    }

    free(Nk_i);
    free(kred_i);
    free(kpt_displs);
    free(displs_all);

    if (pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0) {
            free(recvbuf_eig);
        }
    }

    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            free(eig_all);
        }
    }
    if (!rank)
        printf("\nFinish writing eigenvalues ...\n");
}

void Calculate_MMN(SPARC_OBJ *pSPARC, int num_kpts, int nntot, int *nnlist,
                   int *nncell, int num_bands,
                   double complex MMN_Matrix[][nntot][num_bands][num_bands]) {

    int rank;
    int size;

    double complex *orbital_global = NULL;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (!rank)
        orbital_global = (double complex *)malloc(
            pSPARC->Nd * pSPARC->Nstates * pSPARC->Nkpts_sym * pSPARC->Nspin *
            pSPARC->Nspinor_eig * sizeof(double complex));

    Collect_orbital(pSPARC, orbital_global);

    // print all orbitals for DEBUG

    if (!rank)
        printf("\nStart calculating MMN ...\n");
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    // calculate WANNIER MMN MATRIX

    if (!rank) {

        double a1[3];
        double a2[3];
        double a3[3];

        double Lx = pSPARC->latvec_scale_x;
        double Ly = pSPARC->latvec_scale_y;
        double Lz = pSPARC->latvec_scale_z;

        a1[0] = pSPARC->LatVec[0] * Lx;
        a1[1] = pSPARC->LatVec[1] * Lx;
        a1[2] = pSPARC->LatVec[2] * Lx;
        a2[0] = pSPARC->LatVec[3] * Ly;
        a2[1] = pSPARC->LatVec[4] * Ly;
        a2[2] = pSPARC->LatVec[5] * Ly;
        a3[0] = pSPARC->LatVec[6] * Lz;
        a3[1] = pSPARC->LatVec[7] * Lz;
        a3[2] = pSPARC->LatVec[8] * Lz;

        double b1[3];
        double b2[3];
        double b3[3];
        double volume = 0.0;
        double crossx, crossy, crossz;

        Cross_Product(&crossx, &crossy, &crossz, a2[0], a2[1], a2[2], a3[0],
                      a3[1], a3[2]);

        volume = crossx * a1[0] + crossy * a1[1] + crossz * a1[2];
        // Calculate b1
        Cross_Product(&crossx, &crossy, &crossz, a2[0], a2[1], a2[2], a3[0],
                      a3[1], a3[2]);
        b1[0] = 2.0 * M_PI * crossx / (volume);
        b1[1] = 2.0 * M_PI * crossy / (volume);
        b1[2] = 2.0 * M_PI * crossz / (volume);

        // Calculate b2
        Cross_Product(&crossx, &crossy, &crossz, a3[0], a3[1], a3[2], a1[0],
                      a1[1], a1[2]);
        b2[0] = 2.0 * M_PI * crossx / (volume);
        b2[1] = 2.0 * M_PI * crossy / (volume);
        b2[2] = 2.0 * M_PI * crossz / (volume);

        // Calculate b3
        Cross_Product(&crossx, &crossy, &crossz, a1[0], a1[1], a1[2], a2[0],
                      a2[1], a2[2]);
        b3[0] = 2.0 * M_PI * crossx / (volume);
        b3[1] = 2.0 * M_PI * crossy / (volume);
        b3[2] = 2.0 * M_PI * crossz / (volume);

        for (int spin = 0; spin < pSPARC->Nspin; spin++) {
            for (int kpt = 0; kpt < num_kpts; kpt++) {
                double kcart[3] = {
                    pSPARC->k1[kpt] * b1[0] + pSPARC->k2[kpt] * b2[0] +
                        pSPARC->k3[kpt] * b3[0],
                    pSPARC->k1[kpt] * b1[1] + pSPARC->k2[kpt] * b2[1] +
                        pSPARC->k3[kpt] * b3[1],
                    pSPARC->k1[kpt] * b1[2] + pSPARC->k2[kpt] * b2[2] +
                        pSPARC->k3[kpt] * b3[2]};
                for (int nn = 0; nn < nntot; nn++) {

                    int image_index = nnlist[kpt * nntot + nn] - 1;

                    double kpcart[3] = {pSPARC->k1[image_index] * b1[0] +
                                            pSPARC->k2[image_index] * b2[0] +
                                            pSPARC->k3[image_index] * b3[0],
                                        pSPARC->k1[image_index] * b1[1] +
                                            pSPARC->k2[image_index] * b2[1] +
                                            pSPARC->k3[image_index] * b3[1],
                                        pSPARC->k1[image_index] * b1[2] +
                                            pSPARC->k2[image_index] * b2[2] +
                                            pSPARC->k3[image_index] * b3[2]};

                    int n1 = nncell[0 * num_kpts * nntot + kpt * nntot + nn];
                    int n2 = nncell[1 * num_kpts * nntot + kpt * nntot + nn];
                    int n3 = nncell[2 * num_kpts * nntot + kpt * nntot + nn];

                    double bcart[3] = {kpcart[0] - kcart[0],
                                       kpcart[1] - kcart[1],
                                       kpcart[2] - kcart[2]};

                    for (int m = 0; m < num_bands; m++) {
                        for (int n = 0; n < num_bands; n++) {
                            for (int s = 0; s < pSPARC->Nspinor_eig; s++) {
                                // MMN matrix element between band m and n at
                                // kpt and image_index
                                double complex mmn_element = 0.0 + 0.0 * I;
                                for (int iz = 0; iz < pSPARC->Nx; iz++) {
                                    double gz = (pSPARC->Nz > 1)
                                                    ? ((double)iz /
                                                       (double)(pSPARC->Nz - 1))
                                                    : 0.0;
                                    for (int iy = 0; iy < pSPARC->Ny; iy++) {
                                        double gy =
                                            (pSPARC->Nz > 1)
                                                ? ((double)iy /
                                                   (double)(pSPARC->Ny - 1))
                                                : 0.0;
                                        for (int ix = 0; ix < pSPARC->Nx;
                                             ix++) {
                                            double gx =
                                                (pSPARC->Nx > 1)
                                                    ? ((double)iz /
                                                       (double)(pSPARC->Nz - 1))
                                                    : 0.0;

                                            double r[3];
                                            r[0] = gx * a1[0] + gy * a2[0] +
                                                   gz * a3[0];
                                            r[1] = gx * a1[1] + gy * a2[1] +
                                                   gz * a3[1];
                                            r[2] = gx * a1[2] + gy * a2[2] +
                                                   gz * a3[2];
                                            int i =
                                                iz * pSPARC->Nx * pSPARC->Ny +
                                                iy * pSPARC->Nx + ix;

                                            double phi = bcart[0] * r[0] +
                                                         bcart[1] * r[1] +
                                                         bcart[2] * r[2];

                                            double complex psi =
                                                cos(phi) + I * sin(phi);
                                            // pSPARC->Xorb_kpt[];
                                            mmn_element +=
                                                conj(
                                                    orbital_global
                                                        [kpt * pSPARC->Nstates *
                                                             pSPARC->Nspin *
                                                             pSPARC->Nd *
                                                             pSPARC
                                                                 ->Nspinor_eig +
                                                         spin *
                                                             pSPARC->Nstates *
                                                             pSPARC->Nd *
                                                             pSPARC
                                                                 ->Nspinor_eig +
                                                         m * pSPARC->Nd *
                                                             pSPARC
                                                                 ->Nspinor_eig +
                                                         s * pSPARC->Nd + i]) *
                                                psi *
                                                orbital_global
                                                    [image_index *
                                                         pSPARC->Nstates *
                                                         pSPARC->Nspin *
                                                         pSPARC->Nd *
                                                         pSPARC->Nspinor_eig +
                                                     spin * pSPARC->Nstates *
                                                         pSPARC->Nd *
                                                         pSPARC->Nspinor_eig +
                                                     n * pSPARC->Nd *
                                                         pSPARC->Nspinor_eig +
                                                     s * pSPARC->Nd + i];
                                        }
                                    }
                                }
                                mmn_element *= pSPARC->dV;
                                // mmn_element /= pSPARC->Nd; // Normalize
                                MMN_Matrix[spin * pSPARC->Nkpts_sym *
                                               pSPARC->Nspinor_eig +
                                           kpt * pSPARC->Nspinor_eig +
                                           s][nn][m][n] = mmn_element;
                            }
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

void Calculate_AMN(SPARC_OBJ *pSPARC, int num_bands, int num_kpts, int num_wann,
                   int *exclude_bands, double *proj_site, int *proj_l,
                   int *proj_m, double *proj_z, double *proj_x,
                   double *proj_zona, int gamma_only, int spionr,
                   double complex *AMN_Matrix) {
    int rank;
    int size;

    double complex *orbital_global = NULL;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    if (!rank) {
        printf("\nStart calculating AMN ... \n");

        orbital_global = (double complex *)malloc(
            pSPARC->Nd * pSPARC->Nstates * pSPARC->Nkpts_sym * pSPARC->Nspin *
            pSPARC->Nspinor_eig * sizeof(double complex));
    }

    Collect_orbital(pSPARC, orbital_global);
    // calculate WANNIER AMN MATRIX

    double a1[3], a2[3], a3[3];
    {
        double Lx = pSPARC->latvec_scale_x;
        double Ly = pSPARC->latvec_scale_y;
        double Lz = pSPARC->latvec_scale_z;
        a1[0] = pSPARC->LatVec[0] * Lx;
        a1[1] = pSPARC->LatVec[1] * Lx;
        a1[2] = pSPARC->LatVec[2] * Lx;
        a2[0] = pSPARC->LatVec[3] * Ly;
        a2[1] = pSPARC->LatVec[4] * Ly;
        a2[2] = pSPARC->LatVec[5] * Ly;
        a3[0] = pSPARC->LatVec[6] * Lz;
        a3[1] = pSPARC->LatVec[7] * Lz;
        a3[2] = pSPARC->LatVec[8] * Lz;
    }

    int Nx = pSPARC->Nx;
    int Ny = pSPARC->Ny;
    int Nz = pSPARC->Nz;
    int Nd = pSPARC->Nd;
    double dV = pSPARC->dV;

    /* 填充 AMN：只在 rank 0 上计算，其他 rank 只做接收或最终广播 */
    if (!rank) {
        for (int kpt = 0; kpt < num_kpts; ++kpt) {
            for (int band = 0; band < num_bands; ++band) {

                const int excluded =
                    (exclude_bands != NULL && exclude_bands[band] != 0);

                for (int iw = 0; iw < num_wann; ++iw) {

                    double complex accum = 0.0 + 0.0 * I;

                    if (!excluded) {
                        /* 投影中心分数坐标 */
                        const double fx = proj_site[3 * iw + 0];
                        const double fy = proj_site[3 * iw + 1];
                        const double fz = proj_site[3 * iw + 2];

                        /* 量子数与指数参数 */
                        /* const int l = proj_l[iw]; */
                        /* const int m = proj_m[iw]; */

                        int l = proj_l[iw];
                        int m = proj_m[iw];

                        if (l < 0) {
                            l = -l;
                            m = cubic_index_to_m(l, m);
                        }

                        double zona = (proj_zona ? proj_zona[iw] : 1.0);
                        if (!(zona > 0.0) || !isfinite(zona))
                            zona = 1.0;

                        /* 使用 proj_z / proj_x 构造局部坐标系 */
                        double zdir[3] = {0.0, 0.0, 1.0};
                        double xdir[3] = {1.0, 0.0, 0.0};
                        if (proj_z) {
                            zdir[0] = proj_z[3 * iw + 0];
                            zdir[1] = proj_z[3 * iw + 1];
                            zdir[2] = proj_z[3 * iw + 2];
                        }
                        if (proj_x) {
                            xdir[0] = proj_x[3 * iw + 0];
                            xdir[1] = proj_x[3 * iw + 1];
                            xdir[2] = proj_x[3 * iw + 2];
                        }

                        double e1[3], e2[3], e3[3];
                        build_local_frame_from_zx(zdir, xdir, e1, e2, e3);

                        /* 选择径向模型：
                                                      1 => 高斯型 R_l = r^l
                           exp(-zona r^2) 0 => Slater 型 R_l = r^l exp(-zona r)
                         */
                        const int use_gaussian = 1;

                        /* 积分截断半径，加速用（经验阈值） */
                        double rcut =
                            (use_gaussian) ? (4.0 / sqrt(zona)) : (8.0 / zona);
                        if (!(rcut > 0.0) || !isfinite(rcut))
                            rcut = 1e9;

                        /* 对 spin 与 spinor
                         * 分量求和（如需分自旋写文件，可在此拆分）
                         */
                        // double complex psi_sum = 0.0 + 0.0 * I;
                        for (int spin = 0; spin < pSPARC->Nspin; ++spin) {
                            for (int s = 0; s < pSPARC->Nspinor_eig; ++s) {

                                for (int iz = 0; iz < Nz; ++iz) {
                                    double gz =
                                        (Nz > 1)
                                            ? ((double)iz / (double)(Nz - 1))
                                            : 0.0;
                                    double dzf = wrap_mhalf_half(gz - fz);

                                    for (int iy = 0; iy < Ny; ++iy) {
                                        double gy = (Ny > 1)
                                                        ? ((double)iy /
                                                           (double)(Ny - 1))
                                                        : 0.0;
                                        double dyf = wrap_mhalf_half(gy - fy);

                                        for (int ix = 0; ix < Nx; ++ix) {
                                            double gx = (Nx > 1)
                                                            ? ((double)ix /
                                                               (double)(Nx - 1))
                                                            : 0.0;
                                            double dxf =
                                                wrap_mhalf_half(gx - fx);

                                            double rglob[3];
                                            rglob[0] = dxf * a1[0] +
                                                       dyf * a2[0] +
                                                       dzf * a3[0];
                                            rglob[1] = dxf * a1[1] +
                                                       dyf * a2[1] +
                                                       dzf * a3[1];
                                            rglob[2] = dxf * a1[2] +
                                                       dyf * a2[2] +
                                                       dzf * a3[2];

                                            double r;

                                            r = sqrt(rglob[0] * rglob[0] +
                                                     rglob[1] * rglob[1] +
                                                     rglob[2] * rglob[2]);

                                            if (r > rcut)
                                                continue;

                                            /* 旋转到局部轴系：r_loc = [e1 e2
                                             * e3]^T rglob */
                                            double rloc[3];

                                            rloc[0] = e1[0] * rglob[0] +
                                                      e1[1] * rglob[1] +
                                                      e1[2] * rglob[2];
                                            rloc[1] = e2[0] * rglob[0] +
                                                      e2[1] * rglob[1] +
                                                      e2[2] * rglob[2];
                                            rloc[2] = e3[0] * rglob[0] +
                                                      e3[1] * rglob[1] +
                                                      e3[2] * rglob[2];

                                            /* φ(r) = R_l(r) Y_lm( r̂_loc ) */
                                            double Ylm = Ylm_real_from_cart(
                                                l, m, rloc[0], rloc[1],
                                                rloc[2]);
                                            double Rl =
                                                use_gaussian
                                                    ? radial_gaussian(l, r,
                                                                      zona)
                                                    : radial_slater(l, r, zona);
                                            double phi_proj = Rl * Ylm;

                                            int gindex =
                                                iz * Ny * Nx + iy * Nx + ix;

                                            int base = kpt * pSPARC->Nstates *
                                                           pSPARC->Nspin * Nd *
                                                           pSPARC->Nspinor_eig +
                                                       spin * pSPARC->Nstates *
                                                           Nd *
                                                           pSPARC->Nspinor_eig +
                                                       band * Nd *
                                                           pSPARC->Nspinor_eig +
                                                       s * Nd;
                                            double complex psi =
                                                orbital_global[base + gindex];
                                            accum += conj(psi) * (phi_proj)*dV;
                                        }
                                    }
                                }

                                AMN_Matrix[(spin + s) * num_kpts * num_bands *
                                               num_wann +
                                           kpt * num_bands * num_wann +
                                           band * num_wann + iw] = accum;
                            }
                        }
                    } /* !excluded */
                }
            }
        }
    } /* rank 0 */

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (!rank) {
        free(orbital_global);
        printf("\nTime for calculating AMN: %.3f ms\n", (t2 - t1) * 1e3);
    }
#endif
}

void Collect_orbital(SPARC_OBJ *pSPARC, double complex *orbital_global) {

    int gridsizes[3], rank, tag, orbital_flag = 0, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int flag = pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 ||
               pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL;
    MPI_Comm alldmcomm;
    int color = (flag == 1) ? MPI_UNDEFINED : 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &alldmcomm);
    if (flag)
        return;

    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int Nd = pSPARC->Nd;

    double dx = pSPARC->delta_x;
    double dy = pSPARC->delta_y;
    double dz = pSPARC->delta_z;
    double dV = pSPARC->dV;

    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int size_k = DMndsp * pSPARC->Nband_bandcomm;

    int kpt_start = 0;
    int kpt_end = pSPARC->Nkpts_sym - 1;
    int band_start = 0;
    int band_end = pSPARC->Nstates - 1;
    int spin_start = 0;
    int spin_end = pSPARC->Nspin - 1;

    double complex *orbital_single = NULL;

    orbital_single = (double complex *)malloc(Nd * pSPARC->Nspinor_eig *
                                              sizeof(double complex));

    for (int kpt = kpt_start; kpt <= kpt_end; kpt++) {
        int kpt_flag =
            (pSPARC->kpt_start_indx <= kpt && kpt <= pSPARC->kpt_end_indx);
        int kpt_shift = kpt - pSPARC->kpt_start_indx;

        for (int band = band_start; band <= band_end; band++) {
            int band_flag = (pSPARC->band_start_indx <= band &&
                             band <= pSPARC->band_end_indx);
            int band_shift = band - pSPARC->band_start_indx;

            for (int spin = spin_start; spin <= spin_end; spin++) {
                int spin_flag = (pSPARC->spin_start_indx <= spin &&
                                 spin <= pSPARC->spin_end_indx);
                int spin_shift = spin - pSPARC->spin_start_indx;

                tag = kpt * pSPARC->Nstates * pSPARC->Nspin +
                      band * pSPARC->Nspin + spin;
                int rank_comm;
                MPI_Comm_rank(pSPARC->dmcomm, &rank_comm);

                if (kpt_flag && band_flag && spin_flag) {

                    if (pSPARC->isGammaPoint) {
                        Collect_orbital_real(
                            pSPARC,
                            pSPARC->Xorb + band_shift * DMndsp +
                                kpt_shift * size_k + spin_shift * DMnd,
                            gridsizes, pSPARC->DMVertices_dmcomm, pSPARC->dV,
                            pSPARC->Nspinor_eig, spin, kpt, band,
                            pSPARC->dmcomm, orbital_single);
                    } else {
                        Collect_orbital_complex(
                            pSPARC,
                            pSPARC->Xorb_kpt + band_shift * DMndsp +
                                kpt_shift * size_k + spin_shift * DMnd,
                            gridsizes, pSPARC->DMVertices_dmcomm, pSPARC->dV,
                            pSPARC->Nspinor_eig, spin, kpt, band,
                            pSPARC->dmcomm, orbital_single);
                    }

                    if (rank_comm == 0) {
                        if (rank == 0) {
                            memcpy(orbital_global +
                                       kpt * pSPARC->Nstates * pSPARC->Nspin *
                                           Nd * pSPARC->Nspinor_eig +
                                       spin * pSPARC->Nstates * Nd *
                                           pSPARC->Nspinor_eig +
                                       band * Nd * pSPARC->Nspinor_eig,
                                   orbital_single,
                                   sizeof(double complex) *
                                       (Nd * pSPARC->Nspinor_eig));
                        } else {
                            MPI_Send(orbital_single, Nd * pSPARC->Nspinor_eig,
                                     MPI_DOUBLE_COMPLEX, 0, tag, alldmcomm);
                        }
                    }
                }
                if (rank == 0) {
                    if (rank_comm == 0) {
                        if (!kpt_flag || !band_flag || !spin_flag) {
                            MPI_Recv(orbital_global +
                                         kpt * pSPARC->Nstates * pSPARC->Nspin *
                                             Nd * pSPARC->Nspinor_eig +
                                         spin * pSPARC->Nstates * Nd *
                                             pSPARC->Nspinor_eig +
                                         band * Nd * pSPARC->Nspinor_eig,
                                     Nd * pSPARC->Nspinor_eig,
                                     MPI_DOUBLE_COMPLEX, MPI_ANY_SOURCE, tag,

                                     alldmcomm, MPI_STATUS_IGNORE);
                        }
                    }
                }
                MPI_Barrier(alldmcomm);
            }
        }
    }

    // gather all orbital_single to rank 0 of global comm
    if (rank == 0) {
        for (int kpt = 9; kpt < pSPARC->kpt_end_indx; kpt++) {
            int kpt_shift = kpt - pSPARC->kpt_start_indx;
            int kpt_flag =
                kpt <= pSPARC->kpt_start_indx && kpt >= pSPARC->kpt_end_indx;

            for (int band = pSPARC->band_start_indx;
                 band <= pSPARC->band_end_indx; band++) {
                int band_shift = band - pSPARC->band_start_indx;
                int band_flag = band <= pSPARC->band_start_indx &&
                                band >= pSPARC->band_end_indx;

                for (int spin = pSPARC->spin_start_indx;
                     spin <= pSPARC->spin_end_indx; spin++) {
                    int spin_shift = spin - pSPARC->spin_start_indx;
                    int spin_flag = spin <= pSPARC->spin_start_indx &&
                                    spin >= pSPARC->spin_end_indx;

                    int tag = kpt * pSPARC->Nstates * pSPARC->Nspin +
                              band * pSPARC->Nspin + spin;

                    int rank_comm;
                    MPI_Comm_rank(pSPARC->dmcomm, &rank_comm);
                    if (spin_flag && band_flag && kpt_flag) {
                        MPI_Recv(orbital_global +
                                     kpt * pSPARC->Nstates * pSPARC->Nspin *
                                         Nd * pSPARC->Nspinor_eig +
                                     spin * pSPARC->Nstates * Nd *
                                         pSPARC->Nspinor_eig +
                                     band * Nd * pSPARC->Nspinor_eig,
                                 Nd * pSPARC->Nspinor_eig, MPI_DOUBLE_COMPLEX,
                                 MPI_ANY_SOURCE, tag, alldmcomm,
                                 MPI_STATUS_IGNORE);
                    } else {
                        memcpy(orbital_global +
                                   kpt * pSPARC->Nstates * pSPARC->Nspin * Nd *
                                       pSPARC->Nspinor_eig +
                                   spin * pSPARC->Nstates * Nd *
                                       pSPARC->Nspinor_eig +
                                   band * Nd * pSPARC->Nspinor_eig,
                               orbital_single + kpt_shift * band_shift *
                                                    spin_shift *
                                                    (Nd * pSPARC->Nspinor_eig),
                               sizeof(double complex) *
                                   (Nd * pSPARC->Nspinor_eig));
                    }
                }
            }
        }
    }

    // print each rank kpt band spin size
    free(orbital_single);
    MPI_Comm_free(&alldmcomm);
}

void Collect_orbital_real(SPARC_OBJ *pSPARC, double *x, int *gridsizes,
                          int *DMVertices, double dV, int Nspinor_eig,
                          int spin_index, int kpt_index, int band_index,
                          MPI_Comm comm, double complex *orbital_single) {
    if (comm == MPI_COMM_NULL)
        return;

    int nproc_comm, rank_comm;
    MPI_Comm_size(comm, &nproc_comm);
    MPI_Comm_rank(comm, &rank_comm);

    // global size of the vector
    int Nx = gridsizes[0];
    int Ny = gridsizes[1];
    int Nz = gridsizes[2];
    int Nd = Nx * Ny * Nz;
    double complex *x_global = NULL;
    double *x_double = NULL;

    if (rank_comm == 0) {
        x_global =
            (double complex *)malloc(Nd * Nspinor_eig * sizeof(double complex));
        x_double = (double *)malloc(Nd * Nspinor_eig * sizeof(double));
    }

    int DMnx = DMVertices[1] - DMVertices[0] + 1;
    int DMny = DMVertices[3] - DMVertices[2] + 1;
    int DMnz = DMVertices[5] - DMVertices[4] + 1;
    int DMnd = DMnx * DMny * DMnz;

    if (nproc_comm >
        1) { // if there's more than one process, need to collect x first
        int sdims[3], periods[3], my_coords[3];
        MPI_Cart_get(comm, 3, sdims, periods, my_coords);

        /* use DD2DD to collect distributed data */
        // create a cartesian topology on one process (rank 0)
        int rdims[3] = {1, 1, 1}, rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_comm) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int rperiods[3] = {1, 1, 1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, rdims, rperiods, 0, &recv_comm);
        }

        D2D_OBJ d2d_sender, d2d_recvr;
        rDMVert[0] = 0;
        rDMVert[1] = Nx - 1;
        rDMVert[2] = 0;
        rDMVert[3] = Ny - 1;
        rDMVert[4] = 0;
        rDMVert[5] = Nz - 1;

        // set up D2D targets, note that this is time consuming if
        // number of processes is large (> 1000), in that case, do
        // this step only once and keep the d2d target objects
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, DMVertices, rDMVert,
                       comm, sdims, recv_comm, rdims, comm);

        // collect vector to one process
        for (int spinor = 0; spinor < Nspinor_eig; spinor++) {
            D2D(&d2d_sender, &d2d_recvr, gridsizes, DMVertices,
                x + DMnd * spinor, rDMVert, x_double + Nd * spinor, comm, sdims,
                recv_comm, rdims, comm, sizeof(double));
        }

        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, comm, recv_comm);

        if (!rank_comm)
            MPI_Comm_free(&recv_comm);
    } else {
        memcpy(x_double, x, sizeof(double) * Nd * Nspinor_eig);
    }

    // convert double to double complex
    for (int i = 0; i < Nd * Nspinor_eig; i++) {
        x_global[i] = x_double[i] + 0.0 * I;
    }
    free(x_double);

    if (rank_comm == 0) {
        // scale psi to make it L2-norm = 1
        for (int i = 0; i < Nd * Nspinor_eig; i++)
            x_global[i] /= sqrt(dV);
        memcpy(orbital_single, x_global,
               sizeof(double complex) * Nd * Nspinor_eig);
    }

    // free the collected data after printing to file
    if (rank_comm == 0) {
        free(x_global);
    }
}

void Collect_orbital_complex(SPARC_OBJ *pSPARC, double complex *x,
                             int *gridsizes, int *DMVertices, double dV,
                             int Nspinor_eig, int spin_index, int kpt_index,
                             int band_index, MPI_Comm comm,
                             double complex *orbital_single) {
    if (comm == MPI_COMM_NULL)
        return;

    int nproc_comm, rank_comm;
    MPI_Comm_size(comm, &nproc_comm);
    MPI_Comm_rank(comm, &rank_comm);

    // global size of the vector
    int Nx = gridsizes[0];
    int Ny = gridsizes[1];
    int Nz = gridsizes[2];
    int Nd = Nx * Ny * Nz;
    double complex *x_global = NULL;
    double *x_double = NULL;

    if (rank_comm == 0) {
        x_global =
            (double complex *)malloc(Nd * Nspinor_eig * sizeof(double complex));
    }

    int DMnx = DMVertices[1] - DMVertices[0] + 1;
    int DMny = DMVertices[3] - DMVertices[2] + 1;
    int DMnz = DMVertices[5] - DMVertices[4] + 1;
    int DMnd = DMnx * DMny * DMnz;

    if (nproc_comm >
        1) { // if there's more than one process, need to collect x first
        int sdims[3], periods[3], my_coords[3];
        MPI_Cart_get(comm, 3, sdims, periods, my_coords);

        /* use DD2DD to collect distributed data */
        // create a cartesian topology on one process (rank 0)
        int rdims[3] = {1, 1, 1}, rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_comm) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int rperiods[3] = {1, 1, 1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, rdims, rperiods, 0, &recv_comm);
        }

        D2D_OBJ d2d_sender, d2d_recvr;
        rDMVert[0] = 0;
        rDMVert[1] = Nx - 1;
        rDMVert[2] = 0;
        rDMVert[3] = Ny - 1;
        rDMVert[4] = 0;
        rDMVert[5] = Nz - 1;

        // set up D2D targets, note that this is time consuming if
        // number of processes is large (> 1000), in that case, do
        // this step only once and keep the d2d target objects
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, DMVertices, rDMVert,
                       comm, sdims, recv_comm, rdims, comm);

        // collect vector to one process
        for (int spinor = 0; spinor < Nspinor_eig; spinor++) {
            D2D(&d2d_sender, &d2d_recvr, gridsizes, DMVertices,
                x + DMnd * spinor, rDMVert, x_global + Nd * spinor, comm, sdims,
                recv_comm, rdims, comm, sizeof(double _Complex));
        }

        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, comm, recv_comm);

        if (!rank_comm)
            MPI_Comm_free(&recv_comm);
    } else {
        memcpy(x_global, x, sizeof(double complex) * Nd * Nspinor_eig);
    }

    if (rank_comm == 0) {
        // scale psi to make it L2-norm = 1
        for (int i = 0; i < Nd * Nspinor_eig; i++)
            x_global[i] /= sqrt(dV);
        memcpy(orbital_single, x_global,
               sizeof(double complex) * Nd * Nspinor_eig);
    }

    // free the collected data after printing to file
    if (rank_comm == 0) {
        free(x_global);
    }
}

double wrap_mhalf_half(double u) { return u - nearbyint(u); }

void build_local_frame_from_zx(const double zdir_in[3], const double xdir_in[3],
                               double e1[3], double e2[3], double e3[3]) {
    /* 复制输入，避免修改原数组 */
    e3[0] = zdir_in[0];
    e3[1] = zdir_in[1];
    e3[2] = zdir_in[2];
    double xdir[3] = {xdir_in[0], xdir_in[1], xdir_in[2]};

    /* 兜底：若 zdir 近零，则用全局 z=(0,0,1) */
    if (sqrt(e3[0] * e3[0] + e3[1] * e3[1] + e3[2] * e3[2]) < 0.0) {
        e3[0] = 0.0;
        e3[1] = 0.0;
        e3[2] = 1.0;
    }

    /* 去除 xdir 在 zdir 方向的分量，得到与 e3 正交的分量 */
    double proj = xdir[0] * e3[0] + xdir[0] * e3[0] + xdir[0] * e3[0];

    xdir[0] -= proj * e3[0];
    xdir[1] -= proj * e3[1];
    xdir[2] -= proj * e3[2];

    /* 若 xdir 也退化，则选择一个与 e3 正交的默认方向 */
    if (sqrt(xdir[0] * xdir[0] + xdir[1] * xdir[1] + xdir[2] * xdir[2]) < 0.0) {
        /* 选取与 e3 不平行的基向量 */
        double tmp[3] = {1.0, 0.0, 0.0};
        if (fabs(e3[0]) > 0.9) {
            tmp[0] = 0.0;
            tmp[1] = 1.0;
            tmp[2] = 0.0;
        }
        /* e1 = tmp - (tmp·e3) e3 */
        double tproj = tmp[0] * e3[0] + tmp[1] * e3[1] + tmp[2] * e3[2];
        e1[0] = tmp[0] - tproj * e3[0];
        e1[1] = tmp[1] - tproj * e3[1];
        e1[2] = tmp[2] - tproj * e3[2];

        double len = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);

        e1[0] = e1[0] / len;
        e1[1] = e1[1] / len;
        e1[2] = e1[2] / len;
    } else {
        e1[0] = xdir[0];
        e1[1] = xdir[1];
        e1[2] = xdir[2];
    }

    /* e2 = e3 × e1；确保右手系 */
    // vcross(e2, e3, e1);
    e2[0] = e3[1] * e1[2] - e3[2] * e1[1];
    e2[1] = e3[2] * e1[0] - e3[0] * e1[2];
    e2[2] = e3[0] * e1[1] - e3[1] * e1[0];

    double len = sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
    e2[0] = e2[0] / len;
    e2[1] = e2[1] / len;
    e2[2] = e2[2] / len;

    /* 最后再正交一次（数值保险）：e1 = e2 × e3 */

    e1[0] = e2[1] * e3[2] - e2[2] * e3[1];
    e1[1] = e2[2] * e3[0] - e2[0] * e3[2];
    e1[2] = e2[0] * e3[1] - e2[1] * e3[0];

    len = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    e1[0] = e1[0] / len;
    e1[1] = e1[1] / len;
    e1[2] = e1[2] / len;
}

int cubic_index_to_m(int L, int icubic) {
    if (L == 0) {
        return 0; /* 只有一个 s */
    } else if (L == 1) {
        switch (icubic) {
        case 1:
            return +1; /* p_x */
        case 2:
            return -1; /* p_y */
        case 3:
            return 0; /* p_z */
        default:
            break;
        }
    } else if (L == 2) {
        switch (icubic) {
        case 1:
            return -2; /* d_xy */
        case 2:
            return -1; /* d_yz */
        case 3:
            return +1; /* d_zx */
        case 4:
            return +2; /* d_x2-y2 */
        case 5:
            return 0; /* d_3z2-r2 */
        default:
            break;
        }
    }
    /* 回退策略：均匀映射到 [-L..L] */
    int m = icubic - (L + 1); /* 把 1..(2L+1) 映射为 -L..L */
    if (m < -L)
        m = -L;
    if (m > L)
        m = L;
    return m;
}

double Plm(int l, int m, double x) {
    if (m < 0 || m > l)
        return 0.0;

    double pmm = 1.0;
    if (m > 0) {
        double somx2 = sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if (l == m)
        return pmm;

    double pmmp1 = x * (2.0 * m + 1.0) * pmm;
    if (l == m + 1)
        return pmmp1;

    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

double Ylm_real_from_cart(int l, int m, double rx, double ry, double rz) {
    double r = sqrt(rx * rx + ry * ry + rz * rz);
    if (r == 0.0) {
        return (l == 0 && m == 0) ? (1.0 / sqrt(4.0 * M_PI)) : 0.0;
    }
    double costh = rz / r;
    if (costh > 1.0)
        costh = 1.0;
    if (costh < -1.0)
        costh = -1.0;
    double phi = atan2(ry, rx);

    int am = (m >= 0) ? m : -m;

    double log_norm = 0.5 * (log(2.0 * l + 1.0) - log(4.0 * M_PI) +
                             lgamma(l - am + 1) - lgamma(l + am + 1));

    double Nlm = exp(log_norm);
    double Plm_val = Plm(l, am, costh);

    if (m == 0) {
        return Nlm * Plm_val;
    } else if (m > 0) {
        return M_SQRT2 * Nlm * cos(m * phi) * Plm_val;
    } else {
        return M_SQRT2 * Nlm * sin(am * phi) * Plm_val;
    }
}

double radial_gaussian(int l, double r, double alpha) {
    if (alpha <= 0.0)
        alpha = 1.0;
    if (r == 0.0)
        return (l == 0) ? 1.0 : 0.0;
    return pow(r, (double)l) * exp(-alpha * r * r);
}

double radial_slater(int l, double r, double zeta) {
    if (zeta <= 0.0)
        zeta = 1.0;
    if (r == 0.0)
        return (l == 0) ? 1.0 : 0.0;
    return pow(r, (double)l) * exp(-zeta * r);
}
