# NNOpenCL
This is the simplest implementation of Multi Layer Perceptron which train on random data with Stochastic Gradient Descent otpimizer. The implementation is done on GPU with OpenCL.

# Compilation & usage
g++ ./*.cpp -lOpenCL\
./a.out
  
# Data generator
You can customize data generator in main.cpp function called "get_data(x,y)"
