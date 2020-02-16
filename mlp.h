#ifndef MLP_H
#define MLP_H

#include <math.h>
#include "util_cpp.h"
#include "opencl_object.h"

//TODO : put it in another file
#define NB_TENSOR_BUFFER_CAPACITY 100
#define SIZE_TENSOR_BUFFER_CAPACITY 100
#define NB_DATA_BUFFER_CAPACITY 100
#define SIZE_DATA_BUFFER_CAPACITY 100

#define N_INPUT 2
#define N_HIDDEN 5
#define N_OUTPUT 1

class mlp{
private:
    float params[4]={100,10,10,0.001};
    
    void op_unaire(
            const char* function_name,
            float* inputDataA,
            float* result);

    void op_binaire(
            const char* function_name,
            float* matrixA,
            float* matrixB,
            float* result);
    void fit_1batch(
            float list_x[100][100],
            float list_y[100][100],
            float DATA[100][100],
            float * w1,
            float * w2);
public:
    void inference(
            float * x,
            float * label,
            float DATA[100][100],
            float * w1,
            float * w2);


    
    void fit(float list_x[100][100],
            float list_y[100][100],
            int epochs);

};
#endif

