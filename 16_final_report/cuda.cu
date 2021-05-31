#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;

#define M 1024

__global__ void submatmul(float *A, float *B, float *C, int N, int offset) {
  int i = blockIdx.y + offset;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  float sum = 0.0f;
  extern __shared__ float A_s[];
  for (int ks=0; ks<N; ks+=blockDim.x) {
    __syncthreads();
    A_s[threadIdx.x] = A[N*i+ks+threadIdx.x];
    __syncthreads();
    for (int k=ks; k<ks+blockDim.x; k++) {
      sum += A_s[k-ks] * B[N*k+j];
    }
  }
  C[N*i+j] = sum;
}


int main(int argc, char** argv) {
  int size, rank;
  int ndevice;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaGetDeviceCount(&ndevice);

  const int N = 2048;
  if (N % size != 0) {
    if (rank==0)
      cerr << "num of processes must be exponentitation of 2" << endl;
    return -1;
  }
  float *A;
  float *B;
  float *C;
  cudaSetDevice(rank%ndevice);
  cudaMallocManaged(&A,  N * N * sizeof(float));
  cudaMallocManaged(&B,  N * N * sizeof(float));
  cudaMallocManaged(&C,  N * N * sizeof(float));
  float *subC = C + (N*N/size)*rank;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }
  int offset = N/size*rank;
  dim3 grid(N/M, N/size);

  double comp_time = 0, comm_time = 0;
  auto tic = chrono::steady_clock::now();
  submatmul<<<grid,M,M*sizeof(float)>>>(A, B, C, N, offset);
  cudaDeviceSynchronize();
  auto toc = chrono::steady_clock::now();
  comp_time += chrono::duration<double>(toc - tic).count();
  // Communicate
  tic = chrono::steady_clock::now();
  comm_time += chrono::duration<double>(tic - toc).count();
  MPI_Allgather(subC, N*N/size, MPI_FLOAT, C, N*N/size, MPI_FLOAT, MPI_COMM_WORLD);
#pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  MPI_Finalize();
}
