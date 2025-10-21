#ifndef GLOBALS_H
#define GLOBALS_H

// small datasize
#ifdef SMALL
#define NA           7000
#define NONZER       8
#define SHIFT        12
#define NITER        15
#define RCOND        1.0e-1
#define VALID_RESULT 10.362595087124
#endif

// midiumn datasize
#ifdef MEDIUMN
#define NA           14000
#define NONZER       11
#define SHIFT        20
#define NITER        15
#define RCOND        1.0e-1
#define VALID_RESULT 17.130235054029
#endif

// large datasize
#ifdef LARGE
#define NA           75000
#define NONZER       13
#define SHIFT        60
#define NITER        75
#define RCOND        1.0e-1
#define VALID_RESULT 22.712745482631
#endif

#define NZ  (NA * (NONZER + 1) * (NONZER + 1))
#define NAZ (NA * (NONZER + 1))

enum
{
    T_INIT = 0,
    T_BENCH = 1,
    T_CONJ_GRAD = 2,
    T_LAST = 3
};

#endif // GLOBALS_H
