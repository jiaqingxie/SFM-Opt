//
//  main.cpp
//  asl_project
//
//  Created by py X on 2023/4/20.
//

// uncomment if test runtime
#define tsc
// uncomment if output to file for visualization
#define output

#include <iostream>
#include <random>
#include <math.h>
#include <fstream>
#include <cstdlib>
#include "solution.h"
#ifdef tsc
#include "tsc_x86.h"
#endif

#define REP 20
#define CYCLES_REQUIRED 1e8
using namespace std;

Vector2d r,v;

int main(int argc, const char * argv[]) {

    int n_p = atoi(argv[1]); 
    long num_runs = atoi(argv[2]);
    
    // int n_p = 100; 
    // long num_runs = 1000;
    pedestrian pedestrians[n_p];
    Vector2d r_next[n_p], v_next[n_p];
    double total_cycles = 0;
    double cycles = 0.;
    int multiplier = 1;


    for(int i=0; i<n_p; i++) {
        r_next[i] = pedestrians[i].r;
        v_next[i] = pedestrians[i].v;
    }

//   write results to example.txt
    ofstream myfile;
    myfile.open("example.txt");

//    warm up
    #ifdef tsc
    myInt64 start, end;
    do {
            num_runs = num_runs * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < num_runs; i++) {
                pair<Vector2d, Vector2d>(r_next[n_p], v_next[n_p]) = baseline(pedestrians, r_next, v_next, n_p);

                #ifdef tsc
                r = r_next[n_p];
                v = v_next[n_p];
                #endif
                // copy_n(y_bk, n, y);          
            }
            end = stop_tsc(start);

    //         cycles = (double)end;
    //         multiplier = (CYCLES_REQUIRED) / (cycles);
    //         cout<<multiplier<<endl;
            
    } while (multiplier > 2);
    #endif




    for (int iter=0; iter<REP; iter++) {
        #ifdef tsc
        start = start_tsc();
        #endif

//     simulate for num_runs iterations
        for (int j = 0; j < num_runs; j++){
            pair<Vector2d, Vector2d>(r_next[n_p], v_next[n_p]) = baseline(pedestrians, r_next, v_next, n_p);

            r = r_next[n_p];
            v = v_next[n_p];

            #ifdef output
            update(pedestrians, r_next, v_next, myfile, n_p);
            #endif
        }
        #ifdef tsc
        end = stop_tsc(start);
        cycles = ((double)end) / num_runs;
        total_cycles += cycles;
        #endif

    }
    #ifdef tsc
    total_cycles /= REP;
    cycles = total_cycles;
    std::cout<<"Average Number of cycles in total:"<<cycles<<endl;
    #endif

    return 0;
}
