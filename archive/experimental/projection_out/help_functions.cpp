#include "help_functions.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

int loadFile2Memory(const char *filename, char **result)
{
    ifstream stream(filename, ifstream::binary);
    if (!stream) {
        return -1;
    }

    stream.seekg(0, stream.end);
    int size = static_cast<int>(stream.tellg());
    stream.seekg(0, stream.beg);

    *result = new char[size + 1];
    stream.read(*result, size);
    if (!stream) {
        return -2;
    }

    (*result)[size] = 0;
    return size;
}

void run_custom_profiling(int Nb_Of_Kernels,
                          int Nb_Of_Memory_Transfers,
                          cl_event *K_exe_event,
                          cl_event *Mem_op_event,
                          string *list_of_kernel_names)
{
    struct profile_t {
        string action_type;
        string name;
        cl_event event;
        cl_ulong start;
        cl_ulong end;
        double duration_ms;
    };

    profile_t *profile = new profile_t[Nb_Of_Kernels + Nb_Of_Memory_Transfers];

    for (int i = 0; i < Nb_Of_Kernels; i++) {
        profile[i].action_type = "kernel";
        profile[i].name = list_of_kernel_names[i];
        profile[i].event = K_exe_event[i];
    }
    for (int i = 0; i < Nb_Of_Memory_Transfers; i++) {
        profile[Nb_Of_Kernels + i].action_type = "mem (H<->G)";
        profile[Nb_Of_Kernels + i].name = "Transfer_" + to_string(i + 1);
        profile[Nb_Of_Kernels + i].event = Mem_op_event[i];
    }

    cl_ulong app_start = 0;
    cl_ulong app_end = 0;
    cl_ulong kernel_start = 0;
    cl_ulong kernel_end = 0;

    for (int i = 0; i < Nb_Of_Kernels + Nb_Of_Memory_Transfers; i++) {
        cl_int errCode = clGetEventProfilingInfo(profile[i].event, CL_PROFILING_COMMAND_START,
                                                 sizeof(cl_ulong), &profile[i].start, nullptr);
        errCode |= clGetEventProfilingInfo(profile[i].event, CL_PROFILING_COMMAND_END,
                                           sizeof(cl_ulong), &profile[i].end, nullptr);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get profiling info for "
                 << profile[i].name << endl << endl;
            delete[] profile;
            exit(EXIT_FAILURE);
        }

        profile[i].duration_ms = static_cast<double>(profile[i].end - profile[i].start) * 1.0e-6;
        if (app_start == 0 || profile[i].start < app_start) {
            app_start = profile[i].start;
        }
        if (profile[i].end > app_end) {
            app_end = profile[i].end;
        }
        if (profile[i].action_type == "kernel") {
            if (kernel_start == 0 || profile[i].start < kernel_start) {
                kernel_start = profile[i].start;
            }
            if (profile[i].end > kernel_end) {
                kernel_end = profile[i].end;
            }
        }
    }

    cout << "HOST-Info: Profiling summary" << endl;
    cout << "HOST-Info: -------------------------------------------------------------" << endl;
    cout << left << setw(18) << "Type" << setw(24) << "Name" << "Duration (ms)" << endl;
    for (int i = 0; i < Nb_Of_Kernels + Nb_Of_Memory_Transfers; i++) {
        cout << left << setw(18) << profile[i].action_type
             << setw(24) << profile[i].name
             << fixed << setprecision(4) << profile[i].duration_ms << endl;
    }
    cout << "HOST-Info: Kernel elapsed      = "
         << static_cast<double>(kernel_end - kernel_start) * 1.0e-6 << " ms" << endl;
    cout << "HOST-Info: Application elapsed = "
         << static_cast<double>(app_end - app_start) * 1.0e-6 << " ms" << endl;

    delete[] profile;
}
