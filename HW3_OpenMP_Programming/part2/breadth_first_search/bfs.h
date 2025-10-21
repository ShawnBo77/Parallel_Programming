#ifndef BFS_H
#define BFS_H

#include "common/graph.h"

struct solution // NOLINT(readability-identifier-naming): The name is used by shared object;
                // renaming breaks ABI.
{
    int *distances;
};

struct VertexSet
{
    // # of vertices in the set
    int count;
    // max size of buffer vertices
    int max_vertices;
    // array of vertex ids in set
    int *vertices;
};

void bfs_top_down(Graph graph, solution *sol);
void bfs_bottom_up(Graph graph, solution *sol);
void bfs_hybrid(Graph graph, solution *sol);

#endif // BFS_H
