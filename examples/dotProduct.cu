#include "../common/book.h"
#define imin(a, b) (a < b ? a : b) // Macro to find the minimum of two values


const int threadsPerBlock = 256; // Number of threads per block
const int N = 33 * 1024; // Size of the arrays
const int blocksPerGrid = imin(32, (N+threadsPerBlock - 1) / threadsPerBlock); // Number of blocks per grid

__global__ void dot(float *a, float *b, float *c) // Kernel function to add two arrays
{
    __shared__ float cache[threadsPerBlock]; // Shared memory for each block
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the global thread index
    int cacheIndex = threadIdx.x; // Index for the shared memory cache

    float temp = 0; // Temporary variable to store the result
    while (tid < N) // Check if the index is within bounds
    {
        temp += a[tid] * b[tid]; // Perform the multiplication
        tid += blockDim.x * gridDim.x; // Move to the next index
    }

    // Store the result in shared memory
    cache[cacheIndex] = temp; // Store the result in shared memory
    __syncthreads(); // Synchronize threads in the block

    // Reduce the results in shared memory
    int i = blockDim.x / 2; // Half the number of threads
    while (i != 0) // Reduce the results
    {
        if (cacheIndex < i) // Check if the index is within bounds
        {
            cache[cacheIndex] += cache[cacheIndex + i]; // Reduce the results
        }
        __syncthreads(); // Synchronize threads in the block
        i /= 2; // Halve the number of threads
    }

    if (cacheIndex == 0) // If this is the first thread
    {
        c[blockIdx.x] = cache[0]; // Store the result in global memory
    }
}

int main()
{
    float *a, *b, c, *partial_c; // Host arrays
    float *dev_a, *dev_b, *dev_c; // Device arrays

    // Allocate memory on the host
    a = (float*)malloc(N * sizeof(float)); // Allocate memory for N floats on the host
    b = (float*)malloc(N * sizeof(float)); // Allocate memory for N floats on the host
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float)); // Allocate memory for the result on the host

    // Allocate memory on the device
    HANDLE_ERROR(
        cudaMalloc((void**)&dev_a, N * sizeof(float)) // Allocate memory for N floats on the device
    );
    HANDLE_ERROR(
        cudaMalloc((void**)&dev_b, N * sizeof(float)) // Allocate memory for N floats on the device
    );
    HANDLE_ERROR(
        cudaMalloc((void**)&dev_c, blocksPerGrid * sizeof(float)) // Allocate memory for the result on the device
    );

    // Fill the host arrays with data
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy the host arrays to the device
    HANDLE_ERROR(
        cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice) // Copy the host array a to the device array dev_a
    );
    HANDLE_ERROR(
        cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice) // Copy the host array b to the device array dev_b
    );

    // Launch the kernel with N blocks and 1 thread per block
    dot <<< blocksPerGrid, threadsPerBlock >>> (dev_a, dev_b, dev_c); // Launch the kernel with N blocks and 1 thread per block
    
    // Copy the result from the device to the host
    HANDLE_ERROR(
        cudaMemcpy(partial_c, dev_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost) // Copy the device array dev_c to the host array c
    );

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i]; // Sum the partial results 
    }

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf( "TEST CASE -> GPU VALUE %.6g ?= CPU VALUE %.6g\n", c,
                2 * sum_squares( (float)(N - 1) ) );

    // free memory on the GPU side
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
    // free memory on the CPU side
    free( a );
    free( b );
    free( partial_c );
}