#ifndef HELP_FUNCTIONS_H
#define HELP_FUNCTIONS_H

#include <CL/cl.h>
#include <string>

using namespace std;

int loadFile2Memory(const char *filename, char **result);

void run_custom_profiling(int Nb_Of_Kernels,
                          int Nb_Of_Memory_Transfers,
                          cl_event *K_exe_event,
                          cl_event *Mem_op_event,
                          string *list_of_kernel_names);

#endif
