#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <omp.h>
#include <string>

#include <iostream>
#include <sstream>
#include <vector>

#include "bfs.h"
#include "common/cycle_timer.h"
#include "common/graph.h"

constexpr bool USE_BINARY_GRAPH = true;

void reference_bfs_bottom_up(Graph graph, solution *sol);
void reference_bfs_top_down(Graph graph, solution *sol);
void reference_bfs_hybrid(Graph graph, solution *sol);

int main(int argc, char **argv)
{

    int num_threads = -1;
    std::string graph_filename;

    if (argc < 2)
    {
        std::cerr << "Usage: <path/to/graph/file> [num_threads]\n";
        std::cerr << "  To run results for all thread counts: <path/to/graph/file>\n";
        std::cerr << "  Run with a certain number of threads (no correctness run): "
                     "<path/to/graph/file> <num_threads>\n";
        exit(1);
    }

    int thread_count = -1;
    if (argc == 3)
    {
        thread_count = atoi(argv[2]);
    }

    graph_filename = argv[1];

    Graph g;

    printf("----------------------------------------------------------\n");
    printf("Max system threads = %d\n", omp_get_max_threads());
    if (thread_count > 0)
    {
        thread_count = std::min(thread_count, omp_get_max_threads());
        printf("Running with %d threads\n", thread_count);
    }
    printf("----------------------------------------------------------\n");

    printf("Loading graph...\n");
    if (USE_BINARY_GRAPH)
    {
        g = load_graph_binary(graph_filename.c_str());
    }
    else
    {
        g = load_graph(argv[1]);
        printf("storing binary form of graph!\n");
        store_graph_binary(graph_filename.append(".bin").c_str(), g);
        free_graph(g);
        exit(1);
    }
    printf("\n");
    printf("Graph stats:\n");
    printf("  Edges: %d\n", g->num_edges);
    printf("  Nodes: %d\n", g->num_nodes);

    // If we want to run on all threads
    if (thread_count <= -1)
    {
        // Static assignment to get consistent usage across trials
        int max_threads = omp_get_max_threads();

        // static num_threadss
        std::vector<int> num_threads;

        // dynamic num_threads
        for (int i = 1; i < max_threads; i *= 2)
        {
            num_threads.push_back(i);
        }
        num_threads.push_back(max_threads);
        std::size_t n_usage = num_threads.size();

        solution top_sol = {.distances = new int[g->num_nodes]};
        solution bottom_sol = {.distances = new int[g->num_nodes]};
        solution hybrid_sol = {.distances = new int[g->num_nodes]};

        // Solution sphere
        solution ref_sol = {.distances = new int[g->num_nodes]};

        double hybrid_base, top_base, bottom_base;
        double hybrid_time, top_time, bottom_time;

        double ref_hybrid_base, ref_top_base, ref_bottom_base;
        double ref_hybrid_time, ref_top_time, ref_bottom_time;

        double start;
        std::stringstream timing;
        std::stringstream ref_timing;
        std::stringstream relative_timing;

        bool tds_check = true, bus_check = true, hs_check = true;

        timing << "Threads  Top Down          Bottom Up         Hybrid\n";
        ref_timing << "Threads  Top Down          Bottom Up         Hybrid\n";
        relative_timing << "Threads       Top Down          Bottom Up             Hybrid\n";

        // Loop through assignment values;
        for (int i = 0; i < n_usage; i++)
        {
            printf("----------------------------------------------------------\n");
            std::cout << "Running with " << num_threads[i] << " threads" << '\n';
            // Set thread count
            omp_set_num_threads(num_threads[i]);

            // Run implementations
            start = CycleTimer::current_seconds();
            bfs_top_down(g, &top_sol);
            top_time = CycleTimer::current_seconds() - start;

            // Run reference implementation
            start = CycleTimer::current_seconds();
            reference_bfs_top_down(g, &ref_sol);
            ref_top_time = CycleTimer::current_seconds() - start;

            std::cout << "Testing Correctness of Top Down\n";
            for (int j = 0; j < g->num_nodes; j++)
            {
                if (top_sol.distances[j] != ref_sol.distances[j])
                {
                    fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j, top_sol.distances[j],
                            ref_sol.distances[j]);
                    tds_check = false;
                    break;
                }
            }

            // Run implementations
            start = CycleTimer::current_seconds();
            bfs_bottom_up(g, &bottom_sol);
            bottom_time = CycleTimer::current_seconds() - start;

            // Run reference implementation
            start = CycleTimer::current_seconds();
            reference_bfs_bottom_up(g, &ref_sol);
            ref_bottom_time = CycleTimer::current_seconds() - start;

            std::cout << "Testing Correctness of Bottom Up\n";
            for (int j = 0; j < g->num_nodes; j++)
            {
                if (bottom_sol.distances[j] != ref_sol.distances[j])
                {
                    fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j,
                            bottom_sol.distances[j], ref_sol.distances[j]);
                    bus_check = false;
                    break;
                }
            }

            start = CycleTimer::current_seconds();
            bfs_hybrid(g, &hybrid_sol);
            hybrid_time = CycleTimer::current_seconds() - start;

            // Run reference implementation
            start = CycleTimer::current_seconds();
            reference_bfs_hybrid(g, &ref_sol);
            ref_hybrid_time = CycleTimer::current_seconds() - start;

            std::cout << "Testing Correctness of Hybrid\n";
            for (int j = 0; j < g->num_nodes; j++)
            {
                if (hybrid_sol.distances[j] != ref_sol.distances[j])
                {
                    fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j,
                            hybrid_sol.distances[j], ref_sol.distances[j]);
                    hs_check = false;
                    break;
                }
            }

            if (i == 0)
            {
                hybrid_base = hybrid_time;
                ref_hybrid_base = ref_hybrid_time;
                top_base = top_time;
                bottom_base = bottom_time;
                ref_top_base = ref_top_time;
                ref_bottom_base = ref_bottom_time;
            }

            std::array<char, 1024> buf;
            std::array<char, 1024> ref_buf;
            std::array<char, 1024> relative_buf;

            sprintf(buf.data(), "%4d:    %.2f (%.2fx)      %.2f (%.2fx)      %.2f (%.2fx)\n",
                    num_threads[i], top_time, top_base / top_time, bottom_time,
                    bottom_base / bottom_time, hybrid_time, hybrid_base / hybrid_time);
            sprintf(ref_buf.data(), "%4d:    %.2f (%.2fx)      %.2f (%.2fx)      %.2f (%.2fx)\n",
                    num_threads[i], ref_top_time, ref_top_base / ref_top_time, ref_bottom_time,
                    ref_bottom_base / ref_bottom_time, ref_hybrid_time,
                    ref_hybrid_base / ref_hybrid_time);
            sprintf(relative_buf.data(), "%4d:   %14.2f     %14.2f     %14.2f\n", num_threads[i],
                    ref_top_time / top_time, ref_bottom_time / bottom_time,
                    ref_hybrid_time / hybrid_time);

            timing << buf.data();
            ref_timing << ref_buf.data();
            relative_timing << relative_buf.data();
        }

        delete[] top_sol.distances;
        delete[] bottom_sol.distances;
        delete[] hybrid_sol.distances;
        delete[] ref_sol.distances;

        printf("----------------------------------------------------------\n");
        std::cout << "Your Code: Timing Summary" << '\n';
        std::cout << timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "Reference: Timing Summary" << '\n';
        std::cout << ref_timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "Correctness: " << '\n';
        if (!tds_check)
            std::cout << "Top Down Search is not Correct" << '\n';
        if (!bus_check)
            std::cout << "Bottom Up Search is not Correct" << '\n';
        if (!hs_check)
            std::cout << "Hybrid Search is not Correct" << '\n';
        std::cout << '\n' << "Speedup vs. Reference: " << '\n' << relative_timing.str();
    }
    // Run the code with only one thread count and only report speedup
    else
    {
        bool tds_check = true, bus_check = true, hs_check = true;
        solution top_sol = {.distances = new int[g->num_nodes]};
        solution bottom_sol = {.distances = new int[g->num_nodes]};
        solution hybrid_sol = {.distances = new int[g->num_nodes]};

        // Solution sphere
        solution ref_sol = {.distances = new int[g->num_nodes]};

        double hybrid_time, top_time, bottom_time;
        double ref_hybrid_time, ref_top_time, ref_bottom_time;

        double start;
        std::stringstream timing;
        std::stringstream ref_timing;

        timing << "Threads   Top Down    Bottom Up       Hybrid\n";
        ref_timing << "Threads   Top Down    Bottom Up       Hybrid\n";

        // Loop through assignment values;
        std::cout << "Running with " << thread_count << " threads" << '\n';
        // Set thread count
        omp_set_num_threads(thread_count);

        // Run implementations
        start = CycleTimer::current_seconds();
        bfs_top_down(g, &top_sol);
        top_time = CycleTimer::current_seconds() - start;

        // Run reference implementation
        start = CycleTimer::current_seconds();
        reference_bfs_top_down(g, &ref_sol);
        ref_top_time = CycleTimer::current_seconds() - start;

        std::cout << "Testing Correctness of Top Down\n";
        for (int j = 0; j < g->num_nodes; j++)
        {
            if (top_sol.distances[j] != ref_sol.distances[j])
            {
                fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j, top_sol.distances[j],
                        ref_sol.distances[j]);
                tds_check = false;
                break;
            }
        }

        // Run implementations
        start = CycleTimer::current_seconds();
        bfs_bottom_up(g, &bottom_sol);
        bottom_time = CycleTimer::current_seconds() - start;

        // Run reference implementation
        start = CycleTimer::current_seconds();
        reference_bfs_bottom_up(g, &ref_sol);
        ref_bottom_time = CycleTimer::current_seconds() - start;

        std::cout << "Testing Correctness of Bottom Up\n";
        for (int j = 0; j < g->num_nodes; j++)
        {
            if (bottom_sol.distances[j] != ref_sol.distances[j])
            {
                fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j, bottom_sol.distances[j],
                        ref_sol.distances[j]);
                bus_check = false;
                break;
            }
        }

        start = CycleTimer::current_seconds();
        bfs_hybrid(g, &hybrid_sol);
        hybrid_time = CycleTimer::current_seconds() - start;

        // Run reference implementation
        start = CycleTimer::current_seconds();
        reference_bfs_hybrid(g, &ref_sol);
        ref_hybrid_time = CycleTimer::current_seconds() - start;

        std::cout << "Testing Correctness of Hybrid\n";
        for (int j = 0; j < g->num_nodes; j++)
        {
            if (hybrid_sol.distances[j] != ref_sol.distances[j])
            {
                fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j, hybrid_sol.distances[j],
                        ref_sol.distances[j]);
                hs_check = false;
                break;
            }
        }

        std::array<char, 1024> buf;
        std::array<char, 1024> ref_buf;

        sprintf(buf.data(), "%4d:     %8.2f     %8.2f     %8.2f\n", thread_count, top_time,
                bottom_time, hybrid_time);
        sprintf(ref_buf.data(), "%4d:     %8.2f     %8.2f     %8.2f\n", thread_count, ref_top_time,
                ref_bottom_time, ref_hybrid_time);

        timing << buf.data();
        ref_timing << ref_buf.data();

        delete[] top_sol.distances;
        delete[] bottom_sol.distances;
        delete[] hybrid_sol.distances;
        delete[] ref_sol.distances;

        if (!tds_check)
            std::cout << "Top Down Search is not Correct" << '\n';
        if (!bus_check)
            std::cout << "Bottom Up Search is not Correct" << '\n';
        if (!hs_check)
            std::cout << "Hybrid Search is not Correct" << '\n';
        printf("----------------------------------------------------------\n");
        std::cout << "Your Code: Timing Summary" << '\n';
        std::cout << timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "Reference: Timing Summary" << '\n';
        std::cout << ref_timing.str();
        printf("----------------------------------------------------------\n");
    }

    free_graph(g);

    return 0;
}
