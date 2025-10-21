#pragma once

#include <stdbool.h>

#include "globals.h"

//---------------------------------------------------------------------
/* common / main_int_mem / */
int colidx[NZ];
int rowstr[NA + 1];
int iv[NA];
int arow[NA];
int acol[NAZ];

/* common / main_flt_mem / */
double aelt[NAZ];
double a[NZ];
double x[NA + 2];
double z[NA + 2];
double p[NA + 2];
double q[NA + 2];
double r[NA + 2];

/* common / partit_size / */
int naa;
int nzz;
int firstrow;
int lastrow;
int firstcol;
int lastcol;

/* common /urando/ */
double amult;
double tran;

/* common /timers/ */
bool timeron;
//---------------------------------------------------------------------

//---------------------------------------------------------------------
void conj_grad(const int colidx[],
               const int rowstr[],
               const double x[],
               double z[],
               const double a[],
               double p[],
               double q[],
               double r[],
               double *rnorm);
void makea(int n,
           int nz,
           double a[],
           int colidx[],
           int rowstr[],
           int firstrow,
           int lastrow,
           int firstcol,
           int lastcol,
           int arow[],
           int acol[][NONZER + 1],
           double aelt[][NONZER + 1],
           int iv[]);
void sparse(double a[],
            int colidx[],
            int rowstr[],
            int n,
            int nz,
            int nozer,
            const int arow[],
            int acol[][NONZER + 1],
            double aelt[][NONZER + 1],
            int firstrow,
            int lastrow,
            int nzloc[],
            double rcond,
            double shift);
void sprnvc(int n, int nz, int nn1, double v[], int iv[]);
int icnvrt(double x, int ipwr2);
void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
void init(double *zeta);
void iterate(double *zeta, const int *it);
