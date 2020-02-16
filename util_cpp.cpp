
#include "util_cpp.h"



void display_matrix(float* matrix){
    if(matrix[0]>0 and matrix[1]>0){
        uint h=(uint)matrix[0];
        uint w=(uint)matrix[1];
        uint coord=0;
        for(uint i=0;i<h;i++){
            for(uint j=0;j<w;j++){
                coord=i*w+j;
                printf("%f ", matrix[coord+2]);
                if(j==w-1){
                    printf("\n");
                }
            }
        }
    }
}

uint size(float* matrix){
    return (uint) 2*sizeof(float)+(matrix[0]*matrix[1])*sizeof(float);
}

void log_debug(const char* txt) {
    if (LOG_LEVEL == 1) {
        printf("%s", txt);
    }
}

void log_err(const char* txt) {
    perror(txt);
}

bool compare_matrix(float* matrixA, float* matrixB){
    if(matrixA[0]!=matrixB[0] or matrixA[1] != matrixB[1]){
        return false;
    }
    uint size = (uint) matrixA[0]*matrixA[1];
    for(uint i=0;i<size;i++){
        if(matrixA[i+2]!=matrixB[i+2]){
            return false;
        }
    }
    return true;
}