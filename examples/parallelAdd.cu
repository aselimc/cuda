#include "../common/book.h"
#define N 10

__global__ void add(int *a, int *b, int *c) // Kernel function to add two arrays
{
    int index = blockIdx.x; // handle index of the block
    if (index < N) // Check if the index is within bounds
    {
        c[index] = a[index] + b[index]; // Perform the addition
    }
}

int main()
{
    int a[N], b[N], c[N]; // Host arrays
    int *dev_a, *dev_b, *dev_c; // Device arrays

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
    add<<<N, 1>>>(dev_a, dev_b, dev_c); // Launch the kernel with N blocks and 1 thread per block

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
        cudaFree(dev_a) // Free the device array dev_a
    );
    HANDLE_ERROR(
        cudaFree(dev_b) // Free the device array dev_b
    );
    HANDLE_ERROR(
        cudaFree(dev_c) // Free the device array dev_c
    );
    return 0; // Return 0 to indicate success

}