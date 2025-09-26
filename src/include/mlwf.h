#ifndef MLWF_H
#define MLWF_H
#endif

#include "isddft.h"

void Generate_Wannier_Inputs(SPARC_OBJ *pSPARC);

#ifdef __cplusplus
extern "C" {
#endif
void wannier_setup_(const char *seed_name,
                    const int *mp_grid, // int[3]
                    const int *num_kpts,
                    const double *real_lattice,  // double[3*3]
                    const double *recip_lattice, // double[3*3]
                    const double *kpt_latt,      // double[3*num_kpts]
                    const int *num_bands_tot, const int *num_atoms,
                    const char atom_symbols[][3], // char[num_atoms][LEN]
                    const double *atoms_cart,     // double[3*num_atoms]
                    const int *gamma_only, const int *spinors, int *nntot,
                    int *nnlist, // int[num_kpts][num_nnmax]
                    int *nncell, // int[3][num_kpts][num_nnmax]
                    int *num_bands, int *num_wann, double *proj_site,
                    int *proj_l, int *proj_m, int *proj_radial, double *proj_z,
                    double *proj_x, double *proj_zona, int *exclude_bands,
                    int *proj_s,          // optional
                    double *proj_s_qaxis, // optional
                    int seed_name_len, int atom_symbols_len

);
#ifdef __cplusplus
}
#endif

void Calculate_MMN(SPARC_OBJ *pSPARC, int num_kpts, int nntot, int *nnlist,
                   int *nncell, int num_bands,
                   double complex MMN_Matrix[][nntot][num_bands][num_bands]);

void Calculate_AMN(SPARC_OBJ *pSPARC);

void Get_All_Cart_Coord(SPARC_OBJ *pSPARC);
