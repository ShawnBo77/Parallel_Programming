#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "graph.h"

#define GRAPH_HEADER_TOKEN ((int)0xDEADBEEF)

void free_graph(Graph graph)
{
    delete[] graph->outgoing_starts;
    delete[] graph->outgoing_edges;

    delete[] graph->incoming_starts;
    delete[] graph->incoming_edges;
    delete graph;
}

void build_start(graph *graph, const int *scratch)
{
    int num_nodes = graph->num_nodes;
    graph->outgoing_starts = new int[num_nodes];
    for (int i = 0; i < num_nodes; i++)
    {
        graph->outgoing_starts[i] = scratch[i];
    }
}

void build_edges(graph *graph, const int *scratch)
{
    int num_nodes = graph->num_nodes;
    graph->outgoing_edges = new int[graph->num_edges];
    for (int i = 0; i < graph->num_edges; i++)
    {
        graph->outgoing_edges[i] = scratch[num_nodes + i];
    }
}

// Given an outgoing edge adjacency list representation for a directed
// graph, build an incoming adjacency list representation
void build_incoming_edges(graph *graph)
{

    // printf("Beginning build_incoming... (%d nodes)\n", graph->num_nodes);

    int num_nodes = graph->num_nodes;
    int *node_counts = new int[num_nodes];
    int *node_scatter = new int[num_nodes];

    graph->incoming_starts = new int[num_nodes];
    graph->incoming_edges = new int[graph->num_edges];

    for (int i = 0; i < num_nodes; i++)
        node_counts[i] = node_scatter[i] = 0;

    int total_edges = 0;
    // compute number of incoming edges per node
    for (int i = 0; i < num_nodes; i++)
    {
        int start_edge = graph->outgoing_starts[i];
        int end_edge
            = (i == graph->num_nodes - 1) ? graph->num_edges : graph->outgoing_starts[i + 1];
        for (int j = start_edge; j < end_edge; j++)
        {
            int target_node = graph->outgoing_edges[j];
            node_counts[target_node]++;
            total_edges++;
        }
    }
    // printf("Total edges: %d\n", total_edges);
    // printf("Computed incoming edge counts.\n");

    // build the starts array
    graph->incoming_starts[0] = 0;
    for (int i = 1; i < num_nodes; i++)
    {
        graph->incoming_starts[i] = graph->incoming_starts[i - 1] + node_counts[i - 1];
        // printf("%d: %d ", i, graph->incoming_starts[i]);
    }
    // printf("\n");
    // printf("Last edge=%d\n", graph->incoming_starts[num_nodes-1] + node_counts[num_nodes-1]);

    // printf("Computed per-node incoming starts.\n");

    // now perform the scatter
    for (int i = 0; i < num_nodes; i++)
    {
        int start_edge = graph->outgoing_starts[i];
        int end_edge
            = (i == graph->num_nodes - 1) ? graph->num_edges : graph->outgoing_starts[i + 1];
        for (int j = start_edge; j < end_edge; j++)
        {
            int target_node = graph->outgoing_edges[j];
            graph->incoming_edges[graph->incoming_starts[target_node] + node_scatter[target_node]]
                = i;
            node_scatter[target_node]++;
        }
    }

    /*
    // verify
    printf("Verifying graph...\n");

    for (int i=0; i<num_nodes; i++) {
        int outgoing_starts = graph->outgoing_starts[i];
        int end_node = (i == graph->num_nodes-1) ? graph->num_edges : graph->outgoing_starts[i+1];
        for (int j=outgoing_starts; j<end_node; j++) {

            bool verified = false;

            // make sure that i is a neighbor of target_node
            int target_node = graph->outgoing_edges[j];
            int j_start_edge = graph->incoming_starts[target_node];
            int j_end_edge = (target_node == graph->num_nodes-1)
                    ? graph->num_edges :
                    graph->incoming_starts[target_node+1];
            for (int k=j_start_edge; k<j_end_edge; k++) {
                if (graph->incoming_edges[k] == i) {
                    verified = true;
                    break;
                }
            }

            if (!verified) {
                fprintf(stderr,"Error: %d,%d did not verify\n", i, target_node);
            }
        }
    }

    printf("Done verifying\n");
    */

    delete[] node_counts;
    delete[] node_scatter;
}

void get_meta_data(std::ifstream &file, graph *graph)
{
    // going back to the beginning of the file
    file.clear();
    file.seekg(0, std::ios::beg);
    std::string buffer;
    std::getline(file, buffer);
    if ((buffer.compare(std::string("AdjacencyGraph"))))
    {
        std::cout << "Invalid input file" << buffer << '\n';
        exit(1);
    }
    buffer.clear();

    do
    {
        std::getline(file, buffer);
    } while (buffer.empty() || buffer[0] == '#');

    graph->num_nodes = atoi(buffer.c_str());
    buffer.clear();

    do
    {
        std::getline(file, buffer);
    } while (buffer.empty() || buffer[0] == '#');

    graph->num_edges = atoi(buffer.c_str());
}

void read_graph_file(std::ifstream &file, int *scratch)
{
    std::string buffer;
    int idx = 0;
    while (!file.eof())
    {
        buffer.clear();
        std::getline(file, buffer);

        if (!buffer.empty() && buffer[0] == '#')
            continue;

        std::stringstream parse(buffer);
        while (!parse.fail())
        {
            int v;
            parse >> v;
            if (parse.fail())
            {
                break;
            }
            scratch[idx] = v;
            idx++;
        }
    }
}

void print_graph(const graph *graph)
{

    printf("Graph pretty print:\n");
    printf("num_nodes=%d\n", graph->num_nodes);
    printf("num_edges=%d\n", graph->num_edges);

    for (int i = 0; i < graph->num_nodes; i++)
    {

        int start_edge = graph->outgoing_starts[i];
        int end_edge
            = (i == graph->num_nodes - 1) ? graph->num_edges : graph->outgoing_starts[i + 1];
        printf("node %02d: out=%d: ", i, end_edge - start_edge);
        for (int j = start_edge; j < end_edge; j++)
        {
            int target = graph->outgoing_edges[j];
            printf("%d ", target);
        }
        printf("\n");

        start_edge = graph->incoming_starts[i];
        end_edge = (i == graph->num_nodes - 1) ? graph->num_edges : graph->incoming_starts[i + 1];
        printf("         in=%d: ", end_edge - start_edge);
        for (int j = start_edge; j < end_edge; j++)
        {
            int target = graph->incoming_edges[j];
            printf("%d ", target);
        }
        printf("\n");
    }
}

Graph load_graph(const char *filename)
{
    graph *graph = new struct graph;

    // open the file
    std::ifstream graph_file;
    graph_file.open(filename);
    get_meta_data(graph_file, graph);

    int *scratch = new int[graph->num_nodes + graph->num_edges];
    read_graph_file(graph_file, scratch);

    build_start(graph, scratch);
    build_edges(graph, scratch);
    delete[] scratch;

    build_incoming_edges(graph);

    // print_graph(graph);

    return graph;
}

Graph load_graph_binary(const char *filename)
{
    graph *graph = new struct graph;

    FILE *input = fopen(filename, "rb");

    if (!input)
    {
        fprintf(stderr, "Could not open: %s\n", filename);
        exit(1);
    }

    constexpr int header_size = 3;
    std::array<int, header_size> header;

    if (fread(header.data(), sizeof(int), header_size, input) != header_size)
    {
        fprintf(stderr, "Error reading header.\n");
        exit(1);
    }

    if (header[0] != GRAPH_HEADER_TOKEN)
    {
        fprintf(stderr, "Invalid graph file header. File may be corrupt.\n");
        exit(1);
    }

    graph->num_nodes = header[1];
    graph->num_edges = header[2];

    graph->outgoing_starts = new int[graph->num_nodes];
    graph->outgoing_edges = new int[graph->num_edges];

    if (fread(graph->outgoing_starts, sizeof(int), graph->num_nodes, input)
        != (size_t)graph->num_nodes)
    {
        fprintf(stderr, "Error reading nodes.\n");
        exit(1);
    }

    if (fread(graph->outgoing_edges, sizeof(int), graph->num_edges, input)
        != (size_t)graph->num_edges)
    {
        fprintf(stderr, "Error reading edges.\n");
        exit(1);
    }

    fclose(input);

    build_incoming_edges(graph);
    // print_graph(graph);
    return graph;
}

void store_graph_binary(const char *filename, Graph graph)
{

    FILE *output = fopen(filename, "wb");

    if (!output)
    {
        fprintf(stderr, "Could not open: %s\n", filename);
        exit(1);
    }

    std::array<int, 3> header = {
        GRAPH_HEADER_TOKEN,
        graph->num_nodes,
        graph->num_edges,
    };

    if (fwrite(header.data(), sizeof(int), header.size(), output) != header.size())
    {
        fprintf(stderr, "Error writing header.\n");
        exit(1);
    }

    if (fwrite(graph->outgoing_starts, sizeof(int), graph->num_nodes, output)
        != (size_t)graph->num_nodes)
    {
        fprintf(stderr, "Error writing nodes.\n");
        exit(1);
    }

    if (fwrite(graph->outgoing_edges, sizeof(int), graph->num_edges, output)
        != (size_t)graph->num_edges)
    {
        fprintf(stderr, "Error writing edges.\n");
        exit(1);
    }

    fclose(output);
}
