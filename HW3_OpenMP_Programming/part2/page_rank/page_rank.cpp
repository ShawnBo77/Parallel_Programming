#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;
    }

    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/nnodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / nnodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / nnodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }

     */

    double *score_new = new double[nnodes];
    double *contrib = new double[nnodes];
    bool converged = false;
    double global_diff = 0.0;

    while (!converged)
    {
        // dangling (no outgoing edges) sum
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int i = 0; i < nnodes; i++)
        {
            if (outgoing_size(g, i) == 0)
            {
                dangling_sum += solution[i];
            }
        }
        double dangling_contrib = damping * dangling_sum / nnodes;

        // 抽出來做，如果用 sum_incoming += solution[*vj] / outgoing_size(g, *vj) 
        // outgoing_size 會一直重複計算
        #pragma omp parallel for
        for (int i = 0; i < nnodes; ++i) {
            if (outgoing_size(g, i) > 0) {
                contrib[i] = solution[i] / outgoing_size(g, i);
            } else {
                contrib[i] = 0.0;
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < nnodes; i++)
        {
            double sum_incoming = 0.0;

            const Vertex *start = incoming_begin(g, i);
            const Vertex *end = incoming_end(g, i);
            for (const Vertex *vj = start; vj != end; vj++)
            {
                sum_incoming += contrib[*vj];
            }
            
            score_new[i] = damping * sum_incoming + (1.0 - damping) / nnodes + dangling_contrib;
        }

        global_diff = 0.0;
        #pragma omp parallel for reduction(+:global_diff)
        for (int i = 0; i < nnodes; ++i)
        {
            global_diff += std::abs(score_new[i] - solution[i]);
        }
        
        converged = (global_diff < convergence);

        // 更新分數
        #pragma omp parallel for
        for (int i = 0; i < nnodes; i++)
        {
            solution[i] = score_new[i];
        }
    }

    delete[] score_new;
    delete[] contrib;
}
