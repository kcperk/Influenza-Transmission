
#include "globals.h"

__device__ int currentNumberOfNodes = INIT_NUMBER_OF_NODES;
__device__ int numberOfDays = NUMBER_OF_DAYS; 

// Kernel that executes on the CUDA device
__global__ void node(int activeThreads[], int numberOfNeighbors[], int neighborIDs[], int infectionStatus[], int seed)
{
  /* threadIdx represents the ID of the thread */
  int i,j;
  int tx = threadIdx.x;
  int numberOfNeighborsToLookAt;
  int neighborIndex;

  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);
  
  /* continues to loop until the number of days the simulation is set to run */
  while(numberOfDays > 0) {

    /* if thread is not active skip thread activity until everyone else is done */
    if(activeThreads[tx] == 1)
    {
    	if (infectionStatus[tx] != 1)
    	{
	    	/* Determing a random number of neighbors to look at */
	    	numberOfNeighborsToLookAt = curand(&state) % (numberOfNeighbors[tx] + 1);

	    	while( numberOfNeighborsToLookAt > 0 )
	    	{
	    		
	    		if (infectionStatus[tx] != 1)
	    		{
	    			/* Determining a non-negative Neighbor Index */
	    			/* TODO: change non-negative -> unique */
		    		do{
		    			neighborIndex = curand(&state) % numberOfNeighbors[tx];
		    		} while (neighborIndex == -1);

		    		if(infectionStatus[neighborIndex] == 1)
			    	{
			    		infectionStatus[tx] = 1;
			    	}
		   		}

	    		numberOfNeighborsToLookAt--;
	    	}
    	}

    	/* a chance for node deletion */
        if ( (curand(&state) % 100) < 5 )
        {
        	activeThreads[tx] = 0;
        	
        	for(i = 0; i < MAX_NUMBER_OF_NEIGHBORS; i++)
        		neighborIDs[tx*MAX_NUMBER_OF_NEIGHBORS+i] = -1;

        	numberOfNeighbors[tx] = 0;

        }

    }
    __syncthreads();

    printf("\n \n Day %d", NUMBER_OF_DAYS - numberOfDays);

    for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  	   printf("node[%d] = %d \n", i, infectionStatus[i]);
   
    /* a chance for node addition */
    if ( (curand(&state) % 100) < 5 )
    {
    	activeThreads[tx] = 1;
    	
    	numberOfNeighbors[tx] = (curand(&state) % (MAX_NUMBER_OF_NEIGHBORS + 1));

    	for(j = 0; j < MAX_NUMBER_OF_NEIGHBORS; j++)
    	{
     		if( j < numberOfNeighbors[i] )
     		{
       			neighborIDs[i*MAX_NUMBER_OF_NEIGHBORS+j] = curand(&state) % INIT_NUMBER_OF_NODES;
    		} 
        	else 
    		{
        		neighborIDs[i*MAX_NUMBER_OF_NEIGHBORS+j] = -1;
      		}

    	}

    }

    numberOfDays--;
  
  }

}


// main routine that executes on the host
int main(void)
{

  int i,j;

 
  int * hostActiveThreads = (int *) malloc( (MAX_NUMBER_OF_NODES)*(sizeof(int)));
  int * hostNeighborIDs = (int *) malloc( (MAX_NUMBER_OF_NODES*MAX_NUMBER_OF_NEIGHBORS)*(sizeof(int)));
  int * hostNumberOfNeighbors = (int *) malloc(MAX_NUMBER_OF_NODES*(sizeof(int)));
  int * hostInfectionStatus = (int *) malloc(MAX_NUMBER_OF_NODES*(sizeof(int)));

  int * deviceActiveThreads;
  int * deviceNeighborIDs;
  int * deviceNumberOfNeighbors;
  int * deviceInfectionStatus;

  cudaMalloc( (void **) &deviceActiveThreads,(MAX_NUMBER_OF_NODES)* sizeof(int));
  cudaMalloc( (void **) &deviceNeighborIDs,(MAX_NUMBER_OF_NODES*MAX_NUMBER_OF_NEIGHBORS)* sizeof(int));
  cudaMalloc( (void **) &deviceNumberOfNeighbors,(MAX_NUMBER_OF_NODES)* sizeof(int));
  cudaMalloc( (void **) &deviceInfectionStatus,(MAX_NUMBER_OF_NODES)* sizeof(int));


  /* setting initial amount of nodes to be active */
  for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  {
	if (i < INIT_NUMBER_OF_NODES)
        hostActiveThreads[i] = 1;
    else
        hostActiveThreads[i] = 0;

    if(i == 0)
 		hostInfectionStatus[i] = 1;
 	else
 	 	hostInfectionStatus[i] = 0;
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
  cudaMemcpy(deviceInfectionStatus, hostInfectionStatus, (MAX_NUMBER_OF_NODES) * sizeof(int), cudaMemcpyHostToDevice); 


  //for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  // for(j = 0; j < MAX_NUMBER_OF_NEIGHBORS; j++)
  //   printf("[%d][%d] = %d\n", i, j,neighborIDs[i][j]);

  dim3 DimGrid(1,1,1);
  dim3 DimBlock(MAX_NUMBER_OF_NODES,1,1);

  node<<<DimGrid,DimBlock>>>(deviceActiveThreads,deviceNumberOfNeighbors,deviceNeighborIDs,deviceInfectionStatus, time(NULL));

  cudaDeviceSynchronize();

  free(hostNumberOfNeighbors);
  free(hostNeighborIDs);
  free(hostActiveThreads);

  cudaFree(deviceActiveThreads);
  cudaFree(deviceNeighborIDs);
  cudaFree(deviceNumberOfNeighbors);

  return 0;

}
