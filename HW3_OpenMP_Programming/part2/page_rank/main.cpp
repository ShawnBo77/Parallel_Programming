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

#include "common/cycle_timer.h"
#include "common/grade.h"
#include "common/graph.h"

#include "page_rank.h"

constexpr bool USE_BINARY_GRAPH = true;

constexpr float PAGE_RANK_DAMPENING = 0.3f;
constexpr double PAGE_RANK_CONVERGENCE = 1e-7;

void reference_pageRank( // NOLINT(readability-identifier-naming): The name referenced to shared
                         // object; renaming breaks ABI.
    Graph g,
    double *solution,
    double damping,
    double convergence);

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
        // Static num_threads to get consistent usage across trials
        int max_threads = omp_get_max_threads();

        std::vector<int> num_threads;

        // dynamic num_threads
        for (int i = 1; i < max_threads; i *= 2)
        {
            num_threads.push_back(i);
        }
        num_threads.push_back(max_threads);
        std::size_t n_usage = num_threads.size();

        double *sol = new double[g->num_nodes];
        double *ref_sol = new double[g->num_nodes];

        double pagerank_base;
        double pagerank_time;

        double ref_pagerank_base;
        double ref_pagerank_time;

        double start;
        std::stringstream timing;
        std::stringstream ref_timing;
        std::stringstream relative_timing;

        bool pr_check = true;

        timing << "Threads  Time (Speedup)\n";
        ref_timing << "Threads  Time (Speedup)\n";
        relative_timing << "Threads  Speedup\n";

        // Loop through num_threads values;
        for (int i = 0; i < n_usage; i++)
        {
            printf("----------------------------------------------------------\n");
            std::cout << "Running with " << num_threads[i] << " threads" << '\n';
            // Set thread count
            omp_set_num_threads(num_threads[i]);

            // Run implementations
            start = CycleTimer::current_seconds();
            page_rank(g, sol, PAGE_RANK_DAMPENING, PAGE_RANK_CONVERGENCE);
            pagerank_time = CycleTimer::current_seconds() - start;

            // Run staff reference implementation
            start = CycleTimer::current_seconds();
            reference_pageRank(g, ref_sol, PAGE_RANK_DAMPENING, PAGE_RANK_CONVERGENCE);
            ref_pagerank_time = CycleTimer::current_seconds() - start;

            // record single thread times in order to report speedup
            if (num_threads[i] == 1)
            {
                pagerank_base = pagerank_time;
                ref_pagerank_base = ref_pagerank_time;
            }

            std::cout << "Testing Correctness of Page Rank\n";
            if (!compare_approx(g, ref_sol, sol))
            {
                pr_check = false;
            }

            std::array<char, 1024> buf;
            std::array<char, 1024> ref_buf;
            std::array<char, 1024> relative_buf;

            sprintf(buf.data(), "%4d:   %.4f (%.4fx)\n", num_threads[i], pagerank_time,
                    pagerank_base / pagerank_time);
            sprintf(ref_buf.data(), "%4d:   %.4f (%.4fx)\n", num_threads[i], ref_pagerank_time,
                    ref_pagerank_base / ref_pagerank_time);
            sprintf(relative_buf.data(), "%4d:     %.2fx\n", num_threads[i],
                    ref_pagerank_time / pagerank_time);

            timing << buf.data();
            ref_timing << ref_buf.data();
            relative_timing << relative_buf.data();
        }

        delete[] sol;
        delete[] ref_sol;

        printf("----------------------------------------------------------\n");
        std::cout << "Your Code: Timing Summary" << '\n';
        std::cout << timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "Reference: Timing Summary" << '\n';
        std::cout << ref_timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "Correctness: " << '\n';
        if (!pr_check)
            std::cout << "Page Rank is not Correct" << '\n';
        std::cout << '\n' << "Relative Speedup to Reference: " << '\n' << relative_timing.str();
    }
    // Run the code with only one thread count and only report speedup
    else
    {
        bool pr_check = true;

        double pagerank_base;
        double pagerank_time;

        double ref_pagerank_base;
        double ref_pagerank_time;

        double start;
        std::stringstream timing;
        std::stringstream ref_timing;

        timing << "Threads  Time\n";
        ref_timing << "Threads  Time\n";

        // Loop through assignment values;
        std::cout << "Running with " << thread_count << " threads" << '\n';
        // Set thread count
        omp_set_num_threads(thread_count);

        // Run implementations
        double *sol = new double[g->num_nodes];
        start = CycleTimer::current_seconds();
        page_rank(g, sol, PAGE_RANK_DAMPENING, PAGE_RANK_CONVERGENCE);
        pagerank_time = CycleTimer::current_seconds() - start;

        // Run reference implementation
        double *ref_sol = new double[g->num_nodes];
        start = CycleTimer::current_seconds();
        reference_pageRank(g, ref_sol, PAGE_RANK_DAMPENING, PAGE_RANK_CONVERGENCE);
        ref_pagerank_time = CycleTimer::current_seconds() - start;

        std::cout << "Testing Correctness of Page Rank\n";
        if (!compare_approx(g, ref_sol, sol))
        {
            pr_check = false;
        }

        std::array<char, 1024> buf;
        std::array<char, 1024> ref_buf;

        sprintf(buf.data(), "%4d:   %.4f\n", thread_count, pagerank_time);
        sprintf(ref_buf.data(), "%4d:   %.4f\n", thread_count, ref_pagerank_time);

        timing << buf.data();
        ref_timing << ref_buf.data();

        delete[] sol;
        delete[] ref_sol;

        if (!pr_check)
            std::cout << "Page Rank is not Correct" << '\n';
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
