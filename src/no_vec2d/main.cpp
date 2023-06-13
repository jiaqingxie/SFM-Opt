//
//  main.cpp
//  asl_project
//
//  Created by py X on 2023/4/20.
//
#include <list>
#include <iostream>
#include <string>
#include <random>
#include <math.h>
#include <fstream>
#include <cstdlib>
#include "tsc_x86.h"

#define REP 1
#define CYCLES_REQUIRED 1e8
#define EPS 1e-3
#define PERF_TEST
#define UPDATE_FILE // @TODO: rename to MEASURE_ERROR
#define PREFIX "./results/"
#define SUFFIX ".txt"

using namespace std;

typedef void (*comp_func)(int np, int num_runs);

void baseline(int np, int num_runs);
void updateFile(string name);
double compute_error(string filename, int n_p);

void register_functions();
double perf_test(comp_func f, string name, int flops, int n_p, int num_runs);
void add_function(comp_func f, string name);
void init_comp(int n_p);

vector<comp_func> functions;
vector<string> function_names;
vector<int> function_flops;
int num_funcs = 0;

void add_function(comp_func f, string name) {
    int flops = 1;
    functions.push_back(f);
    function_names.push_back(name);
    function_flops.push_back(flops);
    num_funcs++;
}

double perf_test(comp_func f, string name, int flops, int n_p, int num_runs) {
    init_comp(n_p);

    double total_cycles = 0;
    double cycles = 0.;
    int multiplier = 1;

    //    warm up
#ifdef PERF_TEST
    myInt64 start, end;
    init_comp(n_p);
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(n_p, num_runs);
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        // cout << multiplier << endl;

    } while (multiplier > 2);
#endif

    list<double> cycles_list;

    for (int iter = 0; iter < REP; iter++) {
#ifdef PERF_TEST
        start = start_tsc();
#endif

        //     simulate for num_runs iterations
        init_comp(n_p);
        for (int j = 0; j < num_runs; j++) {
            f(n_p, num_runs);
        }
#ifdef PERF_TEST
        end = stop_tsc(start);
        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

        cycles_list.push_back(cycles);
#endif
    }
#ifdef PERF_TEST
    total_cycles /= REP;
    cycles = total_cycles;

    // maybe need to destroy if we use C
    cout << name << " cycles: " << cycles << endl;
    return cycles;
#endif
    return 0.0;
}

int main(int argc, const char* argv[]) {
    int n_p = atoi(argv[1]);
    long num_runs = atoi(argv[2]);
    int i;
    double perf;

    register_functions();

    if (num_funcs == 0) {
        cout << "No functions registered" << endl;
        return 0;
    }

    cout << num_funcs << " functions registered" << endl;

#ifdef UPDATE_FILE
    // start from the second function, since the first one is the baseline
    for (i = 0; i < num_funcs; i++) {
        comp_func f = functions[i];
        init_comp(n_p);
        for (int j = 0; j < num_runs; j++) {
            f(n_p, num_runs);
            updateFile(PREFIX + function_names[i] + SUFFIX);
        }

        double error = compute_error(function_names[i], n_p);
        // cout << "The error is: " << error << endl;
        if (error > EPS) {
            cout << "ERROR: " << function_names[i] << " is no correct, the relative error wrt to the baseline is: " << error << endl;
        }
    }
#endif
#ifdef PERF_TEST
    // measure the performance of each function 
    for (i = 0; i < num_funcs; i++) {
        init_comp(n_p);
        perf = perf_test(functions[i], function_names[i], function_flops[i], n_p, num_runs);
        // #ifdef UPDATE_FILE
        //         updateFile(PREFIX + function_names[i] + SUFFIX);
        // #endif
    }
#endif
    return 0;
}
