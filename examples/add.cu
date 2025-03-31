#include <iostream>
#include "../common/book.h"

__global__ void add(int a, int b, int *c) {
    *c = a + b; // Add a and b and store the result in the memory pointed to by c
}

int main(){
    int c;
    int *dev_c;

    // Allocate memory on the device
    HANDLE_ERROR(
        cudaMalloc(
            (void**) &dev_c, // This is a pointer to the device memory
             sizeof(int) // This is the size of the memory we want to allocate
            )
        );

    add<<<1, 1>>>(2, 7, dev_c); // Launch the the kernel with 1 block and 1 thread


    HANDLE_ERROR(
        cudaMemcpy(
            &c, // This is the destination (host) memory 
            dev_c, // This is the source (device) memory
            sizeof(int), 
            cudaMemcpyDeviceToHost // Runtime: source (device) → destination (host)
            )
        );

    printf("2 + 7 = %d\n", c);
    // Free device memory
    cudaFree(dev_c);
    return 0;
}