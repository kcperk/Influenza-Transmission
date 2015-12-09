#include "globals.h"

__device__ int currentNumberOfNodes = INIT_NUMBER_OF_NODES;
__device__ int numberOfDays = NUMBER_OF_DAYS; 

// Kernel that executes on the CUDA device
__global__ void node(Node * nodeInfoList, int seed)
{
  /* threadIdx represents the ID of the thread */
  int i;
  int tx = threadIdx.x;
  int numberOfNeighborsToLookAt;
  int neighborIndex;

  int storedNodeStatus[MAX_NUMBER_OF_NODES];
  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed+(tx*34), 0, 0, &state);
  
  /* continues to loop until the number of days the simulation is set to run */
  while(numberOfDays > 0) {

  	for(i = 0; i < MAX_NUMBER_OF_NODES;i++)
  	{
  		storedNodeStatus[i] = nodeInfoList[i].nodeStatus;
  	}

  	__syncthreads();

    /* if thread is not active skip thread activity until everyone else is done */
    if(nodeInfoList[tx].isActive == 1)
    {

    	if (nodeInfoList[tx].nodeStatus == UNINFECTED)
    	{
    		/* 0 to Num Neighbors -1 */
	    	numberOfNeighborsToLookAt = curand(&state) % (nodeInfoList[tx].numberOfNeighbors);
	    
	  		while( numberOfNeighborsToLookAt > -1 )
	    	{
	    		if ( nodeInfoList[tx].nodeStatus == UNINFECTED)
	    		{	    
	    			do {
	  					neighborIndex = curand(&state) % INIT_NUMBER_OF_NODES;

	  					if(nodeInfoList[tx].neighborId[neighborIndex] == -1)
	  						neighborIndex = -1;
	      		
	      			} while (neighborIndex == -1);

		    		if(storedNodeStatus[neighborIndex] == INFECTIOUS || storedNodeStatus[neighborIndex] == ASYMPT)
			    	{
			    		if(curand(&state) % 100 < 90)
			    		{
			    			nodeInfoList[tx].nodeStatus = LATENT;
			    			nodeInfoList[tx].dayInfected = NUMBER_OF_DAYS - numberOfDays;
			    		} 

			    		if(curand(&state) % 100 < 10)
			    		{
			    			nodeInfoList[tx].nodeStatus = INCUBATION;
			    			nodeInfoList[tx].dayInfected = NUMBER_OF_DAYS - numberOfDays;
			    		} 

			    	}
		   		}

	    		numberOfNeighborsToLookAt--;
	    	}
    	}

    }

    __syncthreads();

    if ( tx == 0 ) {

    printf("\n \n Day %d \n", NUMBER_OF_DAYS - numberOfDays);

    for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  	{
  		printf("Node %d is %d", i, nodeInfoList[i].nodeStatus);
  		switch(nodeInfoList[i].nodeStatus)
  		{
  			case UNINFECTED:
  				printf(" is uninfected. \n");
			break;
			case LATENT:
				printf(" is latent. \n");
			break;
			case INCUBATION:
				printf(" is incubating. \n");
			break;
			case INFECTIOUS:
				printf(" is infectious. \n");
			break;
			case ASYMPT:
				printf(" is asymptotic. \n");
			break;
			case RECOVERED:
				printf(" is recovered. \n");
			break;
			default:
				printf(" in invalid. \n");
			break;
  		}
  	}

   }

    numberOfDays--;

    if(nodeInfoList[tx].isActive == 1)
    {

		switch(nodeInfoList[tx].nodeStatus)
		{	
			case UNINFECTED:

			break;
			case LATENT:

				if((NUMBER_OF_DAYS - numberOfDays) - nodeInfoList[tx].dayInfected >= 2)
				{
					//printf("sdfsdf\n");
					nodeInfoList[tx].nodeStatus = INFECTIOUS;	
				}

			break;
			case INCUBATION:

				if((NUMBER_OF_DAYS - numberOfDays) - nodeInfoList[tx].dayInfected >= 1)
				{
					nodeInfoList[tx].nodeStatus = ASYMPT;	
				}

			break;
			case INFECTIOUS:

				if((NUMBER_OF_DAYS - numberOfDays) - nodeInfoList[tx].dayInfected >= 5)
				{
					if( curand(&state) % 100 < (((NUMBER_OF_DAYS - numberOfDays) - nodeInfoList[tx].dayInfected)-5)*10 + 70)
						nodeInfoList[tx].nodeStatus = RECOVERED;	
				}

			break;
			case ASYMPT:

				if((NUMBER_OF_DAYS - numberOfDays) - nodeInfoList[tx].dayInfected >= 3)
				{
					if( curand(&state) % 100 < (((NUMBER_OF_DAYS - numberOfDays) - nodeInfoList[tx].dayInfected)-3)*10 + 70)
						nodeInfoList[tx].nodeStatus = RECOVERED;	
				}

			break;
			case RECOVERED:

			break;
			default:


			break;
		}

	}

	__syncthreads();

  }

}


__global__ void initGraph(Node * nodeInfoList, int seed) {

	int tx = threadIdx.x;
	curandState_t state;
	int j, index;

  	curand_init(seed+(tx*56), 0, 0, &state);

  	if(tx < INIT_NUMBER_OF_NODES) {
  		nodeInfoList[tx].isActive = 1;
  		//from 0 to max
  		nodeInfoList[tx].numberOfNeighbors = (curand(&state) % (MAX_NUMBER_OF_NEIGHBORS + 1));

  		for(j = 0; j < MAX_NUMBER_OF_NODES; j++)
  			nodeInfoList[tx].neighborId[j] = -1;

  		for(j = 0; j < MAX_NUMBER_OF_NEIGHBORS; j++)
	    {
	  		do {
	  			index = curand(&state) % INIT_NUMBER_OF_NODES;

	  			if(nodeInfoList[tx].neighborId[index] != -1)
	  				index = -1;
	      		
	      	} while (index == -1);

	      	nodeInfoList[tx].neighborId[index] = 1;
	    } 

  	} else {
  		nodeInfoList[tx].isActive = 0;
  		nodeInfoList[tx].numberOfNeighbors = -1;
  	}

  	if(tx == 0)
  	{
  		nodeInfoList[tx].nodeStatus = LATENT;
  		nodeInfoList[tx].dayInfected = 0;

  	} else {

  		nodeInfoList[tx].nodeStatus = UNINFECTED;
  		nodeInfoList[tx].dayInfected = -1;

  	}


}


// main routine that executes on the host
int main(void)
{

  Node * hostNodeInfoList = (Node *) malloc(MAX_NUMBER_OF_NODES*(sizeof(Node)));
 // int * hostNeighborIDs = (int *) malloc( (MAX_NUMBER_OF_NODES*MAX_NUMBER_OF_NEIGHBORS)*(sizeof(int)));

  Node * deviceNodeInfoList;
//  int * deviceNeighborIDs;
 
  cudaMalloc( (void **) &deviceNodeInfoList,(MAX_NUMBER_OF_NODES)* sizeof(Node));
 // cudaMalloc( (void **) &deviceNeighborIDs,(MAX_NUMBER_OF_NODES*MAX_NUMBER_OF_NEIGHBORS)* sizeof(int));
 
  cudaMemcpy(deviceNodeInfoList, hostNodeInfoList, (MAX_NUMBER_OF_NODES) * sizeof(Node), cudaMemcpyHostToDevice); 

  //for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  // for(j = 0; j < MAX_NUMBER_OF_NEIGHBORS; j++)
  //   printf("[%d][%d] = %d\n", i, j,neighborIDs[i][j]);

  dim3 DimGrid(1,1,1);
  dim3 DimBlock(MAX_NUMBER_OF_NODES,1,1);

  initGraph<<<DimGrid,DimBlock>>>(deviceNodeInfoList, time(NULL));

  cudaDeviceSynchronize();

  node<<<DimGrid,DimBlock>>>(deviceNodeInfoList, time(NULL));

  cudaDeviceSynchronize();

  free(hostNodeInfoList);

  cudaFree(deviceNodeInfoList);

  return 0;

}

