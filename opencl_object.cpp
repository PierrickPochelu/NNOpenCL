#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <CL/cl.h>


#include "opencl_object.h"


opencl_object::opencl_object(){
    
}

void opencl_object::init_kernel(const char* opencl_file, const char* function_name){    
    log_debug("get platform\n");
    get_plateform(platform_id, num_of_platforms);

    log_debug("get gpu\n");
    get_gpu(platform_id, device_id, num_of_devices);

    log_debug("create context on gpu\n");
    clcontext = create_context_on_gpu(platform_id, device_id);

    log_debug("create command queue\n");
    command_queue = create_command_queue(clcontext, device_id);

    log_debug("read file\n");
    char* source_str = read_file(opencl_file, 1e8);

    log_debug("create program with source\n");
    program = get_program_with_source(clcontext, source_str);

    log_debug("build_program\n");
    build_program(program,device_id);

    log_debug("create kernel\n");
    kernel = create_kernel(program,function_name);
}




void opencl_object::put_input_arg(void* input_arg_data,uint input_arg_size){
    uint arg_id=this->args_data.size();
    this->args_data.push_back(input_arg_data);
    this->args_size.push_back(input_arg_size);
    this->args_type.push_back('i');
    
    log_debug("put input args\n");
    cl_mem arg_buffer=put_input(clcontext, kernel, command_queue, input_arg_data, input_arg_size, arg_id);
    
    this->args_cl_mem.push_back(arg_buffer);
}

void opencl_object::put_inout_arg(void* input_arg_data, uint input_arg_size){
    uint arg_id=this->args_data.size();
    this->args_data.push_back(input_arg_data);
    this->args_size.push_back(input_arg_size);
    this->args_type.push_back('b');
    
    log_debug("put input/output args\n");
    cl_mem arg_buffer=put_input(clcontext, kernel, command_queue, input_arg_data, input_arg_size, arg_id);
    
    this->args_cl_mem.push_back(arg_buffer);
}

void opencl_object::put_output_arg(void* output_arg_data,uint output_arg_size){
    uint arg_id=this->args_data.size();
    this->args_data.push_back(output_arg_data);
    this->args_size.push_back(output_arg_size);
    this->args_type.push_back('o');
    
    log_debug("allocate output args\n");
    const cl_mem arg_buffer = allocate_output(clcontext, kernel, output_arg_size, arg_id);
    
    this->args_cl_mem.push_back(arg_buffer);
}

void opencl_object::run_kernel(uint nb_cores){
    log_debug("enqueue kernel + run\n");
    enqueue_kernel(command_queue, kernel, nb_cores);


    log_debug("wait command queue is finished\n");
    clock_t begin = clock();
    clFinish(command_queue);
    clock_t end = clock();
    float elapsed_secs = float(end - begin) / CLOCKS_PER_SEC;
    printf("TIME = %f \n", elapsed_secs);
    
    log_debug("read output\n");
    uint nb_args=args_data.size();
    for(uint i=0;i<nb_args;i++){
        if(args_type[i]=='o' or args_type[i]=='b'){
            read_output(command_queue, args_cl_mem[i], args_size[i],args_data[i]);
        }
    }
}

void opencl_object::release_memory(){
    log_debug("release memory\n");
    for(uint i=0;i<args_cl_mem.size();i++){
        clReleaseMemObject(args_cl_mem[i]);
    }
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(clcontext);
}

opencl_object::~opencl_object(){

}