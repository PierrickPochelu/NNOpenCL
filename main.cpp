#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <CL/cl.h>
#include <string.h>
#include <vector>
#include <ctime> 

#include "mlp.h"

#define NB_TENSOR_BUFFER_CAPACITY 100
#define SIZE_TENSOR_BUFFER_CAPACITY 100
#define NB_DATA_BUFFER_CAPACITY 100
#define SIZE_DATA_BUFFER_CAPACITY 100


void get_data(float x_buffer[NB_DATA_BUFFER_CAPACITY][SIZE_DATA_BUFFER_CAPACITY],
        float y_buffer[NB_DATA_BUFFER_CAPACITY][SIZE_DATA_BUFFER_CAPACITY]
        );

void get_data(float x_buffer[NB_DATA_BUFFER_CAPACITY][SIZE_DATA_BUFFER_CAPACITY],
        float y_buffer[NB_DATA_BUFFER_CAPACITY][SIZE_DATA_BUFFER_CAPACITY]
        ){
    float non_value=-1.1;
    for(uint i=0;i<NB_DATA_BUFFER_CAPACITY;i++){
        x_buffer[i][0]=1;
        x_buffer[i][1]=2;
        x_buffer[i][2]=((float) rand()/RAND_MAX)*2.-1;
        x_buffer[i][3]=((float) rand()/RAND_MAX)*2.-1;
        
        y_buffer[i][0]=1;
        y_buffer[i][1]=1;
        y_buffer[i][2]=x_buffer[i][2];
        
        y_buffer[i][3]=non_value;
        for(uint j=4;j<SIZE_DATA_BUFFER_CAPACITY;j++){
            x_buffer[i][j]=non_value;
            y_buffer[i][j]=non_value;
        }
    }
}





int main(void) {
    float x_buffer[NB_DATA_BUFFER_CAPACITY][SIZE_DATA_BUFFER_CAPACITY];
    float y_buffer[NB_DATA_BUFFER_CAPACITY][SIZE_DATA_BUFFER_CAPACITY];

    get_data(x_buffer,y_buffer);
    
    mlp model=mlp();
    model.fit(x_buffer,y_buffer,100);


    return 0;
}