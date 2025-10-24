#include "bfs.h"

#include <cstdlib>
#include <omp.h>
#include<vector>
#include<cstring>

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> local_frontiers(num_threads);
    // for (int i = 0; i < num_threads; i++) {
    //     local_frontiers[i].reserve(g->num_nodes);
    // }

    // 共享同一組 thread，減少反覆建立 thread pool 的成本
    #pragma omp parallel
    {
        auto &local = local_frontiers[omp_get_thread_num()];
        local.clear();

        #pragma omp for schedule(static)
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge   = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            // 預取下一個 node 的 outgoing 起點，增加 cache 命中率
            if (i + 1 < frontier->count)
                __builtin_prefetch(&g->outgoing_edges[g->outgoing_starts[frontier->vertices[i + 1]]]);

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER &&
                    __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                {
                    local.push_back(outgoing);
                }
            }
        }
    }

    // 合併 local frontier
    int total_new = 0;
    for (int t = 0; t < num_threads; t++)
        total_new += local_frontiers[t].size();

    new_frontier->count = total_new;

    // 合併到 new_frontier->vertices
    int offset = 0;
    for (int t = 0; t < num_threads; t++)
    {
        std::vector<int> &local = local_frontiers[t];
        std::copy(local.begin(), local.end(), new_frontier->vertices + offset);
        offset += local.size();
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bottom_up_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    int current_dist = distances[frontier->vertices[0]];

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> local_frontiers(num_threads);
    // for (int i = 0; i < num_threads; i++) {
    //     local_frontiers[i].reserve(g->num_nodes);
    // }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto &local = local_frontiers[tid];
        local.clear();

        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < g->num_nodes; i++)
        {
            if (distances[i] != NOT_VISITED_MARKER) continue;

            int start_edge = g->incoming_starts[i];
            int end_edge   = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];

            if (i + 1 < g->num_nodes)
                __builtin_prefetch(&g->incoming_edges[g->incoming_starts[i + 1]]);

            for (int edge = start_edge; edge < end_edge; edge++)
            {
                int neighbor = g->incoming_edges[edge];
                if (distances[neighbor] == current_dist)
                {
                    distances[i] = current_dist + 1;
                    local.push_back(i);
                    break;
                }
            }
        }
    }

    int total_new = 0;
    for (int t = 0; t < num_threads; t++) {
        total_new += local_frontiers[t].size();
    }

    new_frontier->count = total_new;

    // 合併到 new_frontier->vertices
    int offset = 0;
    for (int t = 0; t < num_threads; t++)
    {
        std::vector<int> &local = local_frontiers[t];
        std::copy(local.begin(), local.end(), new_frontier->vertices + offset);
        offset += local.size();
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {
        bottom_up_step(graph, frontier, new_frontier, sol->distances);

        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    bool use_top_down = true;

    int alpha = 14; 
    int beta = 24;  

    while (frontier->count != 0)
    {
        int nf = frontier->count;
        int mf = 0;

        int n = graph->num_nodes;

        if (use_top_down && nf > n / alpha) {
            use_top_down = false;
        } else if (!use_top_down && nf < n / beta) {
            use_top_down = true;
        }

        if (use_top_down) {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        } else {
            bottom_up_step(graph, frontier, new_frontier, sol->distances);
        }

        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }


    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}
