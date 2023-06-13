#pragma once
#include <string>
#define UPDATE_FILE
typedef void (*comp_func)(int np, int num_runs);
void add_function(comp_func f, std::string name);
void register_functions();
void baseline();
