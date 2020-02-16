#ifndef UTIL_MATRIX_H
#define UTIL_MATRIX_H
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define LOG_LEVEL 0
void log_debug(const char* txt);
void log_err(const char* txt);

void display_matrix(float* matrix);
uint size(float* matrix);
bool compare_matrix(float* matrixA, float* matrixB); //usefull for unit test
#endif

