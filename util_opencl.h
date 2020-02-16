#ifndef UTIL_OPENCL_H
#define UTIL_OPENCL_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <CL/cl.h>
#include "util_cpp.h"

char* read_file(const char file_name[], int MAX_SOURCE_SIZE);
void get_plateform(cl_platform_id& platform_id, cl_uint& num_of_platforms);
void get_gpu(cl_platform_id& platform_id, cl_device_id& device_id, cl_uint num_of_devices);
const cl_context create_context_on_gpu(cl_platform_id& platform_id, cl_device_id& device_id);
const cl_program get_program_with_source(cl_context& context, char* source_str);
const cl_command_queue create_command_queue(cl_context& context, cl_device_id device_id);
void build_program(cl_program& program,cl_device_id device_id);
const cl_kernel create_kernel(cl_program& program,const char* function_name);
void enqueue_kernel(cl_command_queue& command_queue, cl_kernel& kernel, const int nb_work_items);
cl_mem put_input(cl_context& context, cl_kernel& kernel, cl_command_queue& command_queue,
        void* inputData, uint size, uint arg_pos);
cl_mem put_inout(cl_context& context, cl_kernel& kernel, cl_command_queue& command_queue,
        void* inputData, uint size, uint arg_pos);
cl_mem allocate_output(cl_context& context, cl_kernel& kernel,
        uint size, uint arg_pos);
void read_output(cl_command_queue &command_queue, const cl_mem &output_buffer,
        uint size, void* results);


#endif