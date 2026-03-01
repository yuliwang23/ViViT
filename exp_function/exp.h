#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
// #include <ap_int.h>
// #include <hls_stream.h>
#include <cmath>

float fast_exp(float x);
extern "C"{
void exp_top(float x_in, float *y_out);
}