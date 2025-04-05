#include <iostream>

// alerts compiler that function should compiled on device instead of host
__global__ void kernel (void) { 
}

int main(void)
{
    // angle brackets denote the arguments we plan to pass to runtime system
    // arguments to device code will be passed via parentheses
    kernel<<<1, 1>>>(); 
    printf("Hello, World!\n");
    return 0;
}