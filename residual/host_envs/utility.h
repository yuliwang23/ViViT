#include <CL/cl.h>
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
#include "host.h"


using namespace std;



void init_random_int(int *data, size_t count, int min_value, int max_value);
void *host_aligned_alloc(size_t size_in_bytes);

// loading weight
bool load_txt_values_as_int(const string &filename, int *dst, size_t count);
bool load_txt_values_as_float(const string &filename, float *dst, size_t count);
bool load_txt_values_as_int8(const string &filename, int8_t *dst, size_t count);

// debugging
void print_event_status(const string &label, cl_event event);

// converting variable type
void copy_int_to_float(int in[SEQ_LEN][EMBED_DIM], float out[SEQ_LEN][EMBED_DIM]);
void copy_int_to_float_mp(const int in[SEQ_LEN][EMBED_DIM], float out[SEQ_LEN][EMBED_DIM]);



