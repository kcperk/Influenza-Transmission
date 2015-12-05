
#include "globals.h"

__device__ int currentNumberOfNodes = INIT_NUMBER_OF_NODES;
__device__ int numberOfDays = NUMBER_OF_DAYS; 

// Kernel that executes on the CUDA device
__global__ void node(int activeThreads[], int numberOfNeighbors[])
{
  /* threadIdx represents the ID of the thread */
  int tx = threadIdx.x;

  if(tx == 1)
  {
     printf("ACTIVE THREADS: %d NUMNEIGHS: %d\n", activeThreads[1], numberOfNeighbors[1]);
  }

  /* continues to loop until the number of days the simulation is set to run */
  while(numberOfDays > 0) {

    /* if thread is not active skip thread activity until everyone else is done */
  //  if(activeThreads[tx] == 1)
    //{








    /* a chance for node deletion */
    //    if ( (rand() % 100) < 5 )
    //    activeThreads[tx] = 0;

    //}
    __syncthreads();

    /* TODO: run a chance to detect if node can be born */

    numberOfDays--;
  
  }

}


// main routine that executes on the host
int main(void)
{

  int i,j;

  srand(time(NULL));

  /* array used to determine if thread is active currently or not */
  int * hostActiveThreads = (int *) malloc( (MAX_NUMBER_OF_NODES)*(sizeof(int)));
  int * hostNeighborIDs = (int *) malloc( (MAX_NUMBER_OF_NODES*MAX_NUMBER_OF_NEIGHBORS)*(sizeof(int)));
  int * hostNumberOfNeighbors = (int *) malloc(MAX_NUMBER_OF_NODES*(sizeof(int)));

  int * deviceActiveThreads;
  int * deviceNeighborIDs;
  int * deviceNumberOfNeighbors;

  cudaMalloc(&deviceActiveThreads,(MAX_NUMBER_OF_NODES)* sizeof(int));
  cudaMalloc(&deviceNeighborIDs,(MAX_NUMBER_OF_NODES*MAX_NUMBER_OF_NEIGHBORS)* sizeof(int));
  cudaMalloc(&deviceNumberOfNeighbors,(MAX_NUMBER_OF_NODES)* sizeof(int));

  /* setting initial amount of nodes to be active */
  for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  {
      if (i < INIT_NUMBER_OF_NODES)
        hostActiveThreads[i] = 1;
      else
        hostActiveThreads[i] = 0;
  }

  for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  {

    if( i < INIT_NUMBER_OF_NODES)
      hostNumberOfNeighbors[i] = (rand() % (MAX_NUMBER_OF_NEIGHBORS + 1));
    else 
      hostNumberOfNeighbors[i] = -1;

    for(j = 0; j < MAX_NUMBER_OF_NEIGHBORS; j++)
    {
      if( j < hostNumberOfNeighbors[i] && i < INIT_NUMBER_OF_NODES)
      {
        hostNeighborIDs[i*MAX_NUMBER_OF_NEIGHBORS+j] = rand() % INIT_NUMBER_OF_NODES;
      } 
       else 
      {
        hostNeighborIDs[i*MAX_NUMBER_OF_NEIGHBORS+j] = -1;
      }

    } 

  } 

  /* cudaMemcpy */
  cudaMemcpy(deviceNeighborIDs, hostNeighborIDs, (MAX_NUMBER_OF_NODES * MAX_NUMBER_OF_NEIGHBORS) * sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpy(deviceActiveThreads, hostActiveThreads, (MAX_NUMBER_OF_NODES) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceNumberOfNeighbors, hostNumberOfNeighbors, (MAX_NUMBER_OF_NODES) * sizeof(int), cudaMemcpyHostToDevice); 

  //mainDeviceNumberOfNeighbors = deviceActiveThreads;

  //for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
   // for(j = 0; j < MAX_NUMBER_OF_NEIGHBORS; j++)
   //   printf("[%d][%d] = %d\n", i, j,neighborIDs[i][j]);

  dim3 DimGrid(1,1,1);
  dim3 DimBlock(MAX_NUMBER_OF_NODES,1,1);

  node<<<DimGrid,DimBlock>>>(deviceActiveThreads,deviceNumberOfNeighbors);

  cudaDeviceSynchronize();

  free(hostNumberOfNeighbors);
  free(hostNeighborIDs);
  free(hostActiveThreads);

  //cudaFree(mainDeviceActiveThreads);
  cudaFree(deviceActiveThreads);
  cudaFree(deviceNeighborIDs);
  cudaFree(deviceNumberOfNeighbors);



  return 0;

}
