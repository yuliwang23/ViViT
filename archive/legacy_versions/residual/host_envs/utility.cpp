#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
// openmp
#include <omp.h>
#include "utility.h"

using namespace std;

void init_random_int(int *data, size_t count, int min_value, int max_value)
{
    int range = max_value - min_value + 1;
    for (size_t i = 0; i < count; i++)
    {
        data[i] = min_value + (rand() % range);
    }
}

// Modified from `posix_memalign`
// allocate 4096-byte aligned host memory
// When declaring the host-side matrix (memory), use it
void *host_aligned_alloc(size_t size_in_bytes)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 4096, size_in_bytes))
    {
        return nullptr;
    }
    return ptr;
}

// loading weight
bool load_txt_values_as_int(const string &filename, int *dst, size_t count)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < count; i++)
    {
        float value = 0.0f;
        if (!(fin >> value))
        {
            cerr << "HOST-Error: Failed to parse value " << i << " from " << filename << endl;
            return false;
        }
        dst[i] = static_cast<int>(roundf(value));
    }

    return true;
}

bool load_txt_values_as_float(const string &filename, float *dst, size_t count)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < count; i++)
    {
        if (!(fin >> dst[i]))
        {
            cerr << "HOST-Error: Failed to parse value " << i << " from " << filename << endl;
            return false;
        }
    }

    return true;
}


bool load_txt_values_as_int8(const string &filename, int8_t *dst, size_t count)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < count; i++)
    {
        float value = 0.0f;
        if (!(fin >> value))
        {
            cerr << "HOST-Error: Failed to parse value " << i << " from " << filename << endl;
            return false;
        }
        dst[i] = static_cast<int8_t>(roundf(value));
    }

    return true;
}

// debugging
void print_event_status(const string &label, cl_event event)
{
    if (event == nullptr)
    {
        cout << label << "NULL" << endl;
        return;
    }

    cl_int status = 0;
    cl_int err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
    cout << label;
    if (err != CL_SUCCESS)
    {
        printf("clGetEventInfo ERROR: %d\n", err);
        return;
    }

    switch (status)
    {
    case CL_QUEUED:
        printf("QUEUED\n");
        break;
    case CL_SUBMITTED:
        printf("SUBMITTED\n");
        break;
    case CL_RUNNING:
        printf("RUNNING\n");
        break;
    case CL_COMPLETE:
        printf("COMPLETE\n");
        break;
    default:
        printf("ERROR: %d\n", status);
        break;
    }
}

// converting variable type
void copy_int_to_float(int in[SEQ_LEN][EMBED_DIM], float out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = static_cast<float>(in[i][d]);
}

void copy_int_to_float_mp(
    const int in[SEQ_LEN][EMBED_DIM],
    float out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        #pragma omp simd
        for (int d = 0; d < EMBED_DIM; d++)
        {
            out[i][d] = static_cast<float>(in[i][d]);
        }
    }
}