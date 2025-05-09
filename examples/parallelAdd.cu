#include "../common/book.h"
#define N (33 * 1024)
#define MAX_BLOCKS 1028 // Maximum number of blocks

__global__ void add(int *a, int *b, int *c) // Kernel function to add two arrays
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread index
    while (tid < N) // Check if the index is within bounds
    {
        c[tid] = a[tid] + b[tid]; // Perform the addition
        tid += blockDim.x * gridDim.x; // Move to the next index
    }
}

int main()
{
    int a[N], b[N], c[N]; // Host arrays
    int *dev_a, *dev_b, *dev_c; // Device arrays

    // Define the number of threads per block
    int threadsPerBlock = 256; // Number of threads per block

    // Allocate memory on the device
    HANDLE_ERROR(
        cudaMalloc((void**)&dev_a, N * sizeof(int)) // Allocate memory for N integers on the device
    );
    HANDLE_ERROR(
        cudaMalloc((void**)&dev_b, N * sizeof(int)) // Allocate memory for N integers on the device
    );
    HANDLE_ERROR(
        cudaMalloc((void**)&dev_c, N * sizeof(int)) // Allocate memory for N integers on the device
    );

    // Fill the host arrays with data
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * 2;
    }
    // Copy the host arrays to the device
    HANDLE_ERROR(
        cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice) // Copy the host array a to the device array dev_a
    );
    HANDLE_ERROR(
        cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice) // Copy the host array b to the device array dev_b
    );
    // Launch the kernel with N blocks and 1 thread per block

    // Calculate the number of blocks needed
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Calculate the number of blocks needed

    // If we are going to have arbitrary large arrays, we choose a fixed grid size
    blocksPerGrid = blocksPerGrid > MAX_BLOCKS ? MAX_BLOCKS : blocksPerGrid; // Limit the number of blocks to 128

    add <<< blocksPerGrid, threadsPerBlock >>> (dev_a, dev_b, dev_c); // Launch the kernel with N blocks and 1 thread per block

    // Copy the result from the device to the host
    HANDLE_ERROR(
        cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost) // Copy the device array dev_c to the host array c
    );
    // Print the results
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]); // Print the result of the addition
    }
    // Free the device memory
    HANDLE_ERROR(
        cudaFree(dev_a) // Free the device memory for dev_a
    );
    HANDLE_ERROR(
        cudaFree(dev_b) // Free the device memory for dev_b
    );
    HANDLE_ERROR(
        cudaFree(dev_c) // Free the device memory for dev_c
    );
    return 0; // Return success
}