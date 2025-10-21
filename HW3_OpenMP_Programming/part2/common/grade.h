#ifndef GRADE_H
#define GRADE_H

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <type_traits>
#include <utility>

#include <cfloat>
#include <cmath>

#include <omp.h>

#include "contracts.h"
#include "graph.h"
#include "graph_internal.h"

// Epsilon for approximate float comparisons
constexpr double EPSILON = 0.00000000001;

// Output column size
constexpr int COL_SIZE = 15;

// Point value for apps that are not run.
constexpr int POINTS_NA = -1;

// Point value for apps that yielded incorrect results.
constexpr int POINTS_INCORRECT = -2;

/*
 * Printing functions
 */

static void sep(std::ostream &out, char separator = '-', int length = 78)
{
    for (int i = 0; i < length; i++)
        out << separator;
    out << '\n';
}

static void print_timing_app(std::ostream &timing, const char *app_name)
{
    std::cout << '\n';
    std::cout << "Timing results for " << app_name << ":" << '\n';
    sep(std::cout, '=', 75);

    timing << '\n';
    timing << "Timing results for " << app_name << ":" << '\n';
    sep(timing, '=', 75);
}

/*
 * Correctness checkers
 */

template <class T> bool compare_arrays(Graph graph, T *ref, T *stu)
{
    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (ref[i] != stu[i])
        {
            std::cerr << "*** Results disagree at " << i << " expected " << ref[i] << " found "
                      << stu[i] << '\n';
            return false;
        }
    }
    return true;
}

template <class T> bool compare_approx(Graph graph, T *ref, T *stu)
{
    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (fabs(ref[i] - stu[i]) > EPSILON)
        {
            std::cerr << "*** Results disagree at " << i << " expected " << ref[i] << " found "
                      << stu[i] << '\n';
            return false;
        }
    }
    return true;
}

template <class T> bool compare_arrays_and_display(Graph graph, T *ref, T *stu)
{
    printf("\n----------------------------------\n");
    printf("Visualization of student results");
    printf("\n----------------------------------\n\n");

    int grid_dim = (int)sqrt(graph->num_nodes);
    for (int j = 0; j < grid_dim; j++)
    {
        for (int i = 0; i < grid_dim; i++)
        {
            printf("%02d ", stu[(j * grid_dim) + i]);
        }
        printf("\n");
    }
    printf("\n----------------------------------\n");
    printf("Visualization of reference results");
    printf("\n----------------------------------\n\n");

    grid_dim = (int)sqrt(graph->num_nodes);
    for (int j = 0; j < grid_dim; j++)
    {
        for (int i = 0; i < grid_dim; i++)
        {
            printf("%02d ", ref[(j * grid_dim) + i]);
        }
        printf("\n");
    }

    return compare_arrays<T>(graph, ref, stu);
}

template <class T> bool compare_arrays_and_radius_est(Graph graph, T *ref, T *stu)
{
    bool is_correct = true;
    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (ref[i] != stu[i])
        {
            std::cerr << "*** Results disagree at " << i << " expected " << ref[i] << " found "
                      << stu[i] << '\n';
            is_correct = false;
        }
    }
    int stu_max_val = -1;
    int ref_max_val = -1;
#pragma omp parallel for schedule(dynamic, 512) reduction(max : stu_max_val)
    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (stu[i] > stu_max_val)
            stu_max_val = stu[i];
    }
#pragma omp parallel for schedule(dynamic, 512) reduction(max : ref_max_val)
    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (ref[i] > ref_max_val)
            ref_max_val = ref[i];
    }

    if (ref_max_val != stu_max_val)
    {
        std::cerr << "*** Radius estimates differ. Expected: " << ref_max_val
                  << " Got: " << stu_max_val << '\n';
        is_correct = false;
    }
    return is_correct;
}

#endif /* GRADE_H */
