#ifndef PAGE_RANK_H
#define PAGE_RANK_H

#include "common/graph.h"

void page_rank(Graph g, double *solution, double damping, double convergence);

#endif /* PAGE_RANK_H */
