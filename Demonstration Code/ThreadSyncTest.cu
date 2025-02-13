//Testing properties of thread Synchronization
//nvcc ThreadSyncTest.cu -o ThreadSyncTest

#include <sys/time.h>
#include <stdio.h>

//Length of vectors to be added.
#define N 1000

//Globals
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	BlockSize.x = 1024;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)2*i;	
		B_CPU[i] = (float)i;
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	free(A_CPU); free(B_CPU); free(C_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
}

void additionCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)		
	{		
		c[id] = a[id] + b[id];
	}
}

//This is the kernel. It is the function that will run on the GPU.
//It adds vectors A and B then stores result in vector C
__global__ void AdditionGPU(float *a, float *b, float *c, int n)
{
	int id = threadIdx.x;
	
	if(id < N)
	{
		c[id] = a[id] + b[id];
		//__syncthreads();
	}
	else
	{
		__syncthreads();
	}
	//__syncthreads();
}

int main()
{
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	additionCPU(A_CPU, B_CPU, C_CPU, N);
	
	float sumCPU = 0.0;
	for(int id = 0; id < N; id++)		
	{		
		sumCPU += C_CPU[id];
	}
	
	//Calling the Kernel (GPU) function.	
	AdditionGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	float sumGPU = 0.0;
	for(int id = 0; id < N; id++)		
	{		
		sumGPU += C_CPU[id];
	}

	if(abs(sumCPU - sumGPU) < 0.000001)
	{
		printf("\n Success \n");
	}
	else
	{
		printf("\n Losser \n");
	}
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	return(0);
}
