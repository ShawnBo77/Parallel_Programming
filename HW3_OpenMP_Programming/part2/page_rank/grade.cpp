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

double run_on_graph(graph *g, int num_threads, int num_runs)
{

    double *sol_stu = new double[g->num_nodes];
    double *sol_ref = new double[g->num_nodes];

    omp_set_num_threads(num_threads);

    double start, time;

    // Run implementation
    double stu_time = std::numeric_limits<int>::max();
    for (int r = 0; r < num_runs; r++)
    {
        start = CycleTimer::current_seconds();
        page_rank(g, sol_stu, PAGE_RANK_DAMPENING, PAGE_RANK_CONVERGENCE);
        // reference_pageRank(g, sol_stu, PAGE_RANK_DAMPENING, PAGE_RANK_CONVERGENCE);
        time = CycleTimer::current_seconds() - start;
        stu_time = std::min(stu_time, time);
    }

    // Run reference implementation
    double ref_time = std::numeric_limits<int>::max();
    for (int r = 0; r < num_runs; r++)
    {
        start = CycleTimer::current_seconds();
        reference_pageRank(g, sol_ref, PAGE_RANK_DAMPENING, PAGE_RANK_CONVERGENCE);
        time = CycleTimer::current_seconds() - start;
        ref_time = std::min(ref_time, time);
    }

    bool correct = compare_approx(g, sol_ref, sol_stu);

    delete[] sol_stu;
    delete[] sol_ref;

    if (!correct)
    {
        std::cout << "Page rank incorrect" << '\n';
    }
    else
    {
        std::cout << "ref_time: " << ref_time << "s" << '\n';
        std::cout << "stu_time: " << stu_time << "s" << '\n';
    }

    double max_score = 4;
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

void print_separator_line()
{
    for (int i = 0; i < 43; i++)
    {
        std::cout << "-";
    }
    std::cout << '\n';
}

void print_scores(std::vector<std::string> grade_graphs, std::vector<double> scores)
{

    std::cout.precision(5);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << '\n' << '\n';

    print_separator_line();

    std::cout << "SCORES :" << '\n';

    print_separator_line();

    double total_score = 0.0;

    for (int g = 0; g < grade_graphs.size(); g++)
    {
        auto &graph_name = grade_graphs[g];

        total_score += scores[g];

        std::string max_score = "4";

        std::cout << graph_name;
        for (int i = 0; i < (28 - graph_name.length()); i++)
        {
            std::cout << " ";
        }
        std::cout << "| ";
        std::cout << "  " << scores[g] << " / " << max_score << " |" << '\n';

        print_separator_line();
    }

    std::cout << "TOTAL";
    for (int i = 0; i < (28 - 5); i++)
    {
        std::cout << " ";
    }
    std::cout << "| ";
    std::cout << "  " << total_score << " / "
              << "16"
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
    while ((opt = getopt(argc, argv, "n:r:h")) != EOF)
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

    std::vector<std::string> grade_graphs = {"soc-livejournal1_68m.graph", "com-orkut_117m.graph",
                                             "rmat_200m.graph", "random_500m.graph"};

    std::vector<double> scores(grade_graphs.size());

    int i = 0;
    for (auto &graph_name : grade_graphs)
    {
        graph *g = load_graph(graph_dir + '/' + graph_name);
        std::cout << "\nGraph: " << graph_name << '\n';
        scores[i] = run_on_graph(g, num_threads, num_runs);
        free_graph(g);
        i++;
    }

    print_scores(grade_graphs, scores);

    return 0;
}
