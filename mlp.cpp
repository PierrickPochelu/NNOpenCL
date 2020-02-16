#include "mlp.h"


void mlp::op_unaire(
        const char* function_name,
        float* inputDataA,
        float* result) {
    const char* opencl_file="kernel_source.cl";
    const uint nb_proc=20;
    opencl_object context;
    context.init_kernel(opencl_file, function_name);//we choose what opencl function call
    context.put_input_arg(inputDataA, size(inputDataA));
    context.put_output_arg(result, size(result));
    context.run_kernel(nb_proc);
}


void mlp::op_binaire(
        const char* function_name,
        float* inputDataA,
        float* inputDataB,
        float* result) {
    const char* opencl_file="kernel_source.cl";
    const uint nb_proc=20;
    opencl_object context;
    context.init_kernel(opencl_file, function_name);//we choose what opencl function call
    context.put_input_arg(inputDataA, size(inputDataA));
    context.put_input_arg(inputDataB, size(inputDataB));
    context.put_output_arg(result, size(result));
    context.run_kernel(nb_proc);
    context.release_memory();
}


void mlp::inference(
        float * x,
        float * label,
        float DATA[100][100],
        float * w1,
        float * w2) {
    opencl_object context;
    context.init_kernel("kernel_source.cl", "mlp_inference");//we choose what opencl function call
    context.put_input_arg(x, size(x));
    context.put_input_arg(label, size(label));
    context.put_output_arg(DATA, sizeof(float)*100*100);
    context.put_inout_arg(w1, size(w1));
    context.put_inout_arg(w2, size(w2));
    context.run_kernel(20);
    context.release_memory();
}


void mlp::fit_1batch(
        float list_x[100][100],
        float list_y[100][100],
        float DATA[100][100],
        float * w1,
        float * w2) {
    opencl_object context;
    context.init_kernel("kernel_source.cl", "mlp_training");//we choose what opencl function call
    context.put_input_arg(list_x, sizeof(float)*100*100);
    context.put_input_arg(list_y, sizeof(float)*100*100);
    context.put_inout_arg(DATA, sizeof(float)*100*100);
    context.put_inout_arg(w1, size(w1));
    context.put_inout_arg(w2, size(w2));
    context.put_inout_arg(params, sizeof(float)*4);
    context.run_kernel(20);
    context.release_memory();
}

void mlp::fit(float x_buffer[NB_DATA_BUFFER_CAPACITY][SIZE_DATA_BUFFER_CAPACITY],
        float y_buffer[NB_DATA_BUFFER_CAPACITY][SIZE_DATA_BUFFER_CAPACITY], int epochs){
    float TENSORS_BUFFER[NB_TENSOR_BUFFER_CAPACITY][SIZE_TENSOR_BUFFER_CAPACITY];
    
    /*
    for(uint i=0;i<NB_TENSOR_BUFFER_CAPACITY;i++){
        for(uint j=0;j<SIZE_TENSOR_BUFFER_CAPACITY;j++){
            TENSORS_BUFFER[i][j]=0;
        }
    }
    */
    

     // INIT W1
    float w1[(N_INPUT+1) * N_HIDDEN +2];
    w1[0]=(N_INPUT+1);
    w1[1]=N_HIDDEN;
    for(int i= 2;i<(N_INPUT+1) * N_HIDDEN +2;i++){
        w1[i]=((float)rand()/RAND_MAX)*2; // drawn between -1 and +1
        w1[i]=sqrt(w1[i] / (N_INPUT+1 + N_HIDDEN)); //drawn between -(in+out)/2 and +(in+out)/2
    }
    
    // INIT W2
    float w2[(N_HIDDEN+1)*N_OUTPUT+2];
    w2[0]=N_HIDDEN+1;
    w2[1]=N_OUTPUT;
    for(int i= 2;i<(N_HIDDEN+1)*N_OUTPUT +2;i++){
        w2[i]=((float)rand()/RAND_MAX)*2;
        w2[i]=sqrt(w2[i] / (N_HIDDEN+1 + N_OUTPUT));
    }
    
    
    for (uint i=0; i<epochs; i++){
        display_matrix(x_buffer[i]);
        display_matrix(y_buffer[i]);
        display_matrix(w1);
        fit_1batch(x_buffer, y_buffer, TENSORS_BUFFER, w1, w2);
        display_matrix(TENSORS_BUFFER[5]);
    }

}