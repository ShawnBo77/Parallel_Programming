#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <string>
#include <unistd.h>

#include <iostream>
#include <vector>

#include "../common/cycle_timer.h"
#include "../common/grade.h"
#include "../common/graph.h"
#include "bfs.h"

constexpr bool USE_BINARY_GRAPH = true;

enum BfsType
{
    TOP_DOWN = 0,
    BOTT_UP = 1,
    HYBRID = 2
};

// NOLINTBEGIN(readability-identifier-naming):
// The names are used by shared object; renaming breaks ABI.

void reference_bfs_bottom_up(Graph graph, solution *sol);
void reference_bfs_top_down(Graph graph, solution *sol);
void reference_bfs_hybrid(Graph graph, solution *sol);

// NOLINTEND(readability-identifier-naming)

void usage(const char *binary_name)
{
    std::cout << "Usage: " << binary_name << " [options] graphdir" << '\n';
    std::cout << '\n';
    std::cout << "Options:" << '\n';
    std::cout << "  -n  INT number of threads" << '\n';
    std::cout << "  -r  INT number of runs" << '\n';
    std::cout << "  -h      this commandline help message" << '\n';
}

graph *load_graph(std::string graph_filename)
{
    graph *g;
    if (USE_BINARY_GRAPH)
    {
        g = load_graph_binary(graph_filename.c_str());
    }
    else
    {
        g = load_graph(graph_filename);
        printf("storing binary form of graph!\n");
        store_graph_binary(graph_filename.append(".bin").c_str(), g);
        free_graph(g);
        exit(1);
    }
    return g;
}

double compute_score(bool correct, double ref_time, double stu_time)
{
    double max_score = 1.0;
    double max_perf_score = 0.8 * max_score;
    double correctness_score = 0.2 * max_score;
    correctness_score = (correct) ? correctness_score : 0;

    double ratio = (ref_time / stu_time);

    double slope = max_perf_score / (0.7 - 0.3);
    double offset = 0.3 * slope;

    double perf_score = (correct) ? (ratio * slope) - offset : 0;

    if (perf_score < 0)
        perf_score = 0;
    if (perf_score > max_perf_score)
        perf_score = max_perf_score;

    return (correctness_score + perf_score);
}

void run_on_graph(
    int idx, graph *g, int num_threads, int num_runs, std::vector<std::vector<double>> &scores)
{

    solution ref;
    ref.distances = new int[g->num_nodes];
    solution stu;
    stu.distances = new int[g->num_nodes];

    double start, time;

    omp_set_num_threads(num_threads);

    std::cout << "\nTop down bfs" << '\n';
    double ref_top_down_time = std::numeric_limits<int>::max();
    for (int r = 0; r < num_runs; r++)
    {
        start = CycleTimer::current_seconds();
        reference_bfs_top_down(g, &ref);
        time = CycleTimer::current_seconds() - start;
        ref_top_down_time = std::min(ref_top_down_time, time);
    }

    double stu_top_down_time = std::numeric_limits<int>::max();
    for (int r = 0; r < num_runs; r++)
    {
        start = CycleTimer::current_seconds();
        bfs_top_down(g, &stu);
        // reference_bfs_top_down(g, &stu);
        time = CycleTimer::current_seconds() - start;
        stu_top_down_time = std::min(stu_top_down_time, time);
    }

    bool correct = compare_arrays(g, ref.distances, stu.distances);

    if (!correct)
    {
        std::cout << "Top down bfs incorrect" << '\n';
        std::cout << "ref_time: " << ref_top_down_time << "s" << '\n';
    }
    else
    {
        std::cout << "ref_time: " << ref_top_down_time << "s" << '\n';
        std::cout << "stu_time: " << stu_top_down_time << "s" << '\n';
    }

    scores[idx][TOP_DOWN] = compute_score(correct, ref_top_down_time, stu_top_down_time);

    for (int i = 0; i < g->num_nodes; i++)
    {
        ref.distances[i] = -1;
        stu.distances[i] = -1;
    }

    double ref_bottom_up_time = std::numeric_limits<int>::max();
    for (int r = 0; r < num_runs; r++)
    {
        start = CycleTimer::current_seconds();
        reference_bfs_bottom_up(g, &ref);
        time = CycleTimer::current_seconds() - start;
        ref_bottom_up_time = std::min(ref_bottom_up_time, time);
    }

    std::cout << "\nBottom up bfs" << '\n';
    double stu_bottom_up_time = std::numeric_limits<int>::max();
    for (int r = 0; r < num_runs; r++)
    {
        start = CycleTimer::current_seconds();
        bfs_bottom_up(g, &stu);
        // reference_bfs_bottom_up(g, &stu);
        time = CycleTimer::current_seconds() - start;
        stu_bottom_up_time = std::min(stu_bottom_up_time, time);
    }

    correct = compare_arrays(g, ref.distances, stu.distances);

    if (!correct)
    {
        std::cout << "Bottom up bfs incorrect" << '\n';
        std::cout << "ref_time: " << ref_bottom_up_time << "s" << '\n';
    }
    else
    {
        std::cout << "ref_time: " << ref_bottom_up_time << "s" << '\n';
        std::cout << "stu_time: " << stu_bottom_up_time << "s" << '\n';
    }

    scores[idx][BOTT_UP] = compute_score(correct, ref_bottom_up_time, stu_bottom_up_time);

    for (int i = 0; i < g->num_nodes; i++)
    {
        ref.distances[i] = -1;
        stu.distances[i] = -1;
    }

    std::cout << "\nHybrid bfs" << '\n';

    double ref_hybrid_time = std::numeric_limits<int>::max();
    for (int r = 0; r < num_runs; r++)
    {
        start = CycleTimer::current_seconds();
        reference_bfs_hybrid(g, &ref);
        time = CycleTimer::current_seconds() - start;
        ref_hybrid_time = std::min(ref_hybrid_time, time);
    }

    double stu_hybrid_time = std::numeric_limits<int>::max();
    for (int r = 0; r < num_runs; r++)
    {
        start = CycleTimer::current_seconds();
        bfs_hybrid(g, &stu);
        // reference_bfs_hybrid(g, &stu);
        time = CycleTimer::current_seconds() - start;
        stu_hybrid_time = std::min(stu_hybrid_time, time);
    }

    correct = compare_arrays(g, ref.distances, stu.distances);

    if (!correct)
    {
        std::cout << "Hybrid bfs incorrect" << '\n';
        std::cout << "ref_time: " << ref_hybrid_time << "s" << '\n';
    }
    else
    {
        std::cout << "ref_time: " << ref_hybrid_time << "s" << '\n';
        std::cout << "stu_time: " << stu_hybrid_time << "s" << '\n';
    }

    scores[idx][HYBRID] = compute_score(correct, ref_hybrid_time, stu_hybrid_time);

    delete (stu.distances);
    delete (ref.distances);
}

void print_separator_line()
{
    for (int i = 0; i < 74; i++)
    {
        std::cout << "-";
    }
    std::cout << '\n';
}

void print_scores(std::vector<std::string> grade_graphs, std::vector<std::vector<double>> scores)
{

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << '\n' << '\n';

    print_separator_line();

    std::cout << "SCORES :";
    for (int i = 0; i < (28 - 8); i++)
    {
        std::cout << " ";
    }

    std::cout << "|   Top-Down    |   Bott-Up    |    Hybrid    |" << '\n';

    print_separator_line();

    double total_score = 0.0;
    double top_down_score = 0.0;
    double bottom_up_score = 0.0;
    double hybrid_score = 0.0;
    int max_top_down_score = 0;
    int max_bottom_up_score = 0;
    int max_hybrid_score = 0;

    std::array<int, 3> max_scores_small = {2, 3, 3};
    std::array<int, 3> max_scores_large = {6, 7, 7};
    for (int g = 0; g < grade_graphs.size(); g++)
    {
        auto &graph_name = grade_graphs[g];

        bool small = false;
        if ((graph_name == "grid1000x1000.graph") || (graph_name == "soc-livejournal1_68m.graph")
            || (graph_name == "com-orkut_117m.graph"))
            small = true;

        auto &max_scores = small ? max_scores_small : max_scores_large;

        total_score += scores[g][TOP_DOWN] * max_scores[TOP_DOWN]
                       + scores[g][BOTT_UP] * max_scores[BOTT_UP]
                       + scores[g][HYBRID] * max_scores[HYBRID];
        top_down_score += scores[g][TOP_DOWN] * max_scores[TOP_DOWN];
        bottom_up_score += scores[g][BOTT_UP] * max_scores[BOTT_UP];
        hybrid_score += scores[g][HYBRID] * max_scores[HYBRID];
        max_top_down_score += max_scores[TOP_DOWN];
        max_bottom_up_score += max_scores[BOTT_UP];
        max_hybrid_score += max_scores[HYBRID];

        std::cout << graph_name;
        for (int i = 0; i < (28 - graph_name.length()); i++)
        {
            std::cout << " ";
        }

        std::cout << "| ";
        std::cout << "     " << scores[g][TOP_DOWN] * max_scores[TOP_DOWN] << " / "
                  << max_scores[TOP_DOWN] << " |";
        std::cout << "     " << scores[g][BOTT_UP] * max_scores[BOTT_UP] << " / "
                  << max_scores[BOTT_UP] << " |";
        std::cout << "     " << scores[g][HYBRID] * max_scores[HYBRID] << " / "
                  << max_scores[HYBRID] << " |" << '\n';
        ;

        print_separator_line();
    }

    std::cout << "SUM";
    for (int i = 0; i < (28 - 3); i++)
    {
        std::cout << " ";
    }
    std::cout << "| ";
    std::cout << "   " << top_down_score << " / " << max_top_down_score << " |";
    std::cout << "   " << bottom_up_score << " / " << max_bottom_up_score << " |";
    std::cout << "   " << hybrid_score << " / " << max_hybrid_score << " |" << '\n';
    print_separator_line();

    std::cout << "TOTAL";
    for (int i = 0; i < (59 - 5); i++)
    {
        std::cout << " ";
    }
    std::cout << "|  ";
    std::cout << total_score << " / "
              << "64"
              << " |" << '\n';
    print_separator_line();
}

int main(int argc, char **argv)
{
    int num_threads = omp_get_max_threads();
    int num_runs = 1;
    std::string graph_dir;
    bool grade = false;

    int opt;
    while ((opt = getopt(argc, argv, "n:r:gh")) != EOF)
    {
        switch (opt)
        {
            case 'n':
                num_threads = atoi(optarg);
                break;
            case 'r':
                num_runs = atoi(optarg);
                break;
            case 'h':
            case '?':
            default:
                usage(argv[0]);
                exit(1);
        }
    }

    if (argc <= optind)
    {
        usage(argv[0]);
        exit(1);
    }

    graph_dir = argv[optind];

    printf("Max system threads = %d\n", omp_get_max_threads());
    printf("Running with %d threads\n", num_threads);

    std::vector<std::string> grade_graphs
        = {"grid1000x1000.graph", "soc-livejournal1_68m.graph", "com-orkut_117m.graph",
           "random_500m.graph", "rmat_200m.graph"};

    std::vector<std::vector<double>> scores(grade_graphs.size());
    // top_down 0
    // bott_up 1
    // hybrid 2
    for (int i = 0; i < grade_graphs.size(); i++)
    {
        scores[i] = std::vector<double>(3);
    }

    int idx = 0;
    for (auto &graph_name : grade_graphs)
    {
        graph *g = load_graph(graph_dir + '/' + graph_name);
        std::cout << "\nGraph: " << graph_name << '\n';
        run_on_graph(idx, g, num_threads, num_runs, scores);
        free_graph(g);
        idx++;
    }

    print_scores(grade_graphs, scores);

    return 0;
}
