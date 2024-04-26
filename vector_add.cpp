#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
// CUDA kernel to generate and print random vectors
__global__ void generateAndPrintRandomVectors(curandState *states, int *a, int *b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        curandState state;
        curand_init(clock64() + i, 0, 0, &state); // Initialize RNG for each
        thread
            // Generate random values for vector a and print
            int random_a = curand(&state) % 100;
        printf("a[%d] = %d ", i, random_a);
        a[i] = random_a;
        printf("\n");
        // Generate random values for vector b and print
        int random_b = curand(&state) % 100;
        printf("b[%d] = %d ", i, random_b);
        b[i] = random_b;
        printf("\n");
    }
}
// CUDA kernel to add vectors
__global__ void addVectors(int *a, int *b, int *c, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        c[i] = a[i] + b[i];
    }
    int main()
    {
        // Define vector size
        int size = 10;
        // Allocate memory on host for vectors
        int *h_c = (int *)malloc(size * sizeof(int));
        // Allocate memory on device for vectors
        int *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, size * sizeof(int));
        cudaMalloc(&d_b, size * sizeof(int));
        cudaMalloc(&d_c, size * sizeof(int));
        // Allocate memory for random number generator states
        curandState *d_states;
        cudaMalloc(&d_states, size * sizeof(curandState));
        // Define grid and block sizes
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        // Create CUDA events for time measurement
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // Record start event
        cudaEventRecord(start);
        // Generate and print random vectors on the device
        generateAndPrintRandomVectors<<<blocksPerGrid, threadsPerBlock>>>(d_states, d_a, d_b, size);
        // Launch the kernel to add vectors
        addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
        // Record stop event
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        // Copy result from device to host
        cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
        // Print the result
        printf("Result:\n");
        for (int i = 0; i < size; i++)
        {
            printf("c[%d] = %d", i, h_c[i]);
        }
        printf("\n");
        // Free memory
        free(h_c);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_states);
        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf("Time taken: %f milliseconds\n", milliseconds);
        return 0;
    }