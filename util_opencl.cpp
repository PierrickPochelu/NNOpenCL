#include "util_opencl.h"


char* read_file(const char file_name[], int MAX_SOURCE_SIZE) {
    FILE *fp;
    size_t source_size;

    fp = fopen(file_name, "r");
    if (!fp) {
        log_err("Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    return source_str;
}

void get_plateform(cl_platform_id& platform_id, cl_uint& num_of_platforms) {
    if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) {
        log_err("Unable to get platform_id\n");
        exit(1);
    }
}

void get_gpu(cl_platform_id& platform_id, cl_device_id& device_id, cl_uint num_of_devices) {
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS) {
        log_err("Unable to get device_id");
        exit(1);
    }
}

const cl_context create_context_on_gpu(cl_platform_id& platform_id, cl_device_id& device_id) {
    cl_context_properties properties[3];
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties) platform_id;
    properties[2] = 0; // must be terminated with 0

    cl_int err;
    const cl_context& context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        log_err("cl error : Unable to create context\n");
    }

    return context;
}

const cl_program get_program_with_source(cl_context& context, char* source_str) {
    cl_int err;
    const cl_program& program = clCreateProgramWithSource(context, 1, (const char **) &source_str, NULL, &err);
    if (err != 0) {
        log_err("cl error : Unable to create program with source");
    }
    return program;
}

const cl_command_queue create_command_queue(cl_context& context, cl_device_id device_id) {
    cl_int err;
    const cl_command_queue &command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    return command_queue;
}

void build_program(cl_program& program, cl_device_id device_id) {
    cl_int err=clBuildProgram(program, 0, NULL, NULL, NULL, NULL); //TODO: we can add device
    if (err != CL_SUCCESS) {
        log_err("cl error : Unable to build program");
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("details : %s\n", log);

    }
}

const cl_kernel create_kernel(cl_program& program,const char* function_name) {
    cl_int err;
    const cl_kernel &kernel = clCreateKernel(program, function_name, &err);
    if (err != CL_SUCCESS) {
        log_err("cl error : Error creating kernel");
    }
    return kernel;
}

void enqueue_kernel(cl_command_queue& command_queue, cl_kernel& kernel, const int nb_work_items) {
    //params
    /*
    cl_command_queue command_queue
    cl_kernel kernel
    cl_uint work_dim
    const size_t* global_work_offset
    const size_t* global_work_size
    const size_t* local_work_size
    cl_uint num_events_in_wait_list
    const cl_event* event_wait_list
    cl_event* event
     */
    size_t global_work_size = nb_work_items;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
}

cl_mem put_input(cl_context& context, cl_kernel& kernel, cl_command_queue& command_queue,
        void* inputData, uint size, uint arg_pos) {
    const cl_mem& input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, size, inputData, 0, NULL, NULL);
    clSetKernelArg(kernel, arg_pos, sizeof (cl_mem), &input_buffer);
    return input_buffer;
}

cl_mem put_inout(cl_context& context, cl_kernel& kernel, cl_command_queue& command_queue,
        void* inputData, uint size, uint arg_pos) {
    const cl_mem& input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, size, inputData, 0, NULL, NULL);
    clSetKernelArg(kernel, arg_pos, sizeof (cl_mem), &input_buffer);
    return input_buffer;
}

cl_mem allocate_output(cl_context& context, cl_kernel& kernel, uint size, uint arg_pos) {
    const cl_mem& output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
    clSetKernelArg(kernel, arg_pos, sizeof(cl_mem), &output_buffer);
    return output_buffer;
}

void read_output(cl_command_queue &command_queue, const cl_mem &output_buffer,
        uint size, void* results){
    clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, size, results, 0, NULL, NULL);
    //clReleaseMemObject(output_buffer);
}


