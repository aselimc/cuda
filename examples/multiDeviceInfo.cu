#include "../common/book.h"

int main()
{
    cudaDeviceProp prop; // This is a structure that holds the properties of the device
    int deviceCount;

    // Get the number of devices
    HANDLE_ERROR(
        cudaGetDeviceCount(&deviceCount) // This function returns the number of devices
    );

    // Iterate over all devices
    for (int i=0; i<deviceCount; i++)
    {
        // Get the properties of the device
        HANDLE_ERROR(
            cudaGetDeviceProperties(&prop, i) // This function returns the properties of the device
        );

        // Print the properties of the device
        printf("Device %d: %s\n", i, prop.name);
        printf("  Total global memory: %.0f MB\n", (float)prop.totalGlobalMem / (1024 * 1024));
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);

        if (prop.deviceOverlap)
        {
            printf("  Device supports overlap\n");
        }
        else
        {
            printf("  Device does not support overlap\n");
        }

        if (prop.kernelExecTimeoutEnabled)
        {
            printf("  Device kernel timeout enabled\n");
        }
        else
        {
            printf("  Device kernel timeout disabled\n");
        }

        printf("Memory information:\n");
        printf("Total global memory: %ld\n", prop.totalGlobalMem);
        printf("Total constant memory: %ld\n", prop.totalConstMem);
        printf("Shared memory per block: %ld\n", prop.sharedMemPerBlock);
        printf("Max memory pitch: %ld\n", prop.memPitch);
        printf("Texture alignment: %ld\n", prop.textureAlignment); 
    }


}