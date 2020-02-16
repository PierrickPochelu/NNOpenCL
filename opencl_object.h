#ifndef RUNNING_CONTEXT_H
#define RUNNING_CONTEXT_H
#include <vector>

#include "util_cpp.h"
#include "util_opencl.h"

class opencl_object{
private:
    cl_uint num_of_platforms = 0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_of_devices = 1;
    cl_context clcontext;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    
    std::vector<void*> args_data;
    std::vector<uint> args_size;
    std::vector<char> args_type; //'i' for input, 'o' for output, 'b' for input/output
    std::vector<cl_mem> args_cl_mem;
public :
    opencl_object();
    ~opencl_object();
    
    void init_kernel(const char* opencl_file, const char* function_name);//we choose what opencl function call
    void put_input_arg(void* input_arg_data, uint input_arg_size);
    void put_inout_arg(void* input_arg_data, uint input_arg_size) ;
    void put_output_arg(void* output_arg_data, uint output_arg_size);
    void run_kernel(uint nb_cores);
    void release_memory();
};

#endif

