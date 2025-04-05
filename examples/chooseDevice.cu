#include "../common/book.h"

int main( void ) {
    cudaDeviceProp  prop;
    int dev;

    HANDLE_ERROR( cudaGetDevice( &dev ) );
    printf( "ID of current CUDA device:  %d\n", dev );

    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 3;
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) ); // Returns the ID of the device that matches the properties specified in prop
    printf( "ID of CUDA device closest to revision 1.3:  %d\n", dev );

    HANDLE_ERROR( cudaSetDevice( dev ) );
}
