#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
using namespace std;
// CUDA kernel for matrix multiplication
__global__ void matrixMul(const int *a, const int *b, int *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        int result = 0;
        for (int k = 0; k < N; ++k)
        {
            result += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = result;
    }
}
// Verify result on the CPU
void verifyResult(const vector<int> &a, const vector<int> &b, const vector<int> &c, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int tmp = 0;
            for (int k = 0; k < N; ++k)
            {
                tmp += a[i * N + k] * b[k * N + j];
            }
            assert(tmp == c[i * N + j]);
        }
    }
}

// Output matrix content
void outputMatrix(const vector<int> &matrix, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            cout << matrix[i * N + j] << " ";
        }
        cout << endl;
    }
}
int main()
{
    int N = 8; // Matrix size of 8x8
    // Host vectors
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);
    // Initialize matrices with random values
    srand(time(NULL));
    cout << "Matrix A:" << endl;
    for (int i = 0; i < N * N; ++i)
    {
        h_a[i] = rand() % 100;
        cout << h_a[i] << " ";
        if ((i + 1) % N == 0)
        {
            cout << endl;
        }
    }
    cout << endl;
    cout << "Matrix B:" << endl;
    for (int i = 0; i < N * N; ++i)
    {
        h_b[i] = rand() % 100;
        cout << h_b[i] << " ";
        if ((i + 1) % N == 0)
        {
            cout << endl;
        }
    }
    cout << endl;
    // Allocate device memory
    size_t bytes = N * N * sizeof(int);
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    // Define block and grid dimensions
    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid(1, 1);
    // Create CUDA events for time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record start event
    cudaEventRecord(start);
    // Launch kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Copy result back to host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cout << "Result Matrix:" << endl;
    outputMatrix(h_c, N);
    cout << "Time taken: " << milliseconds << " milliseconds\n";
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}