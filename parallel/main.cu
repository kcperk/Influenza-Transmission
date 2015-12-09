
#include "globals.h"

__device__ int currentNumberOfNodes = INIT_NUMBER_OF_NODES;
__device__ int numberOfDays = NUMBER_OF_DAYS; 

// Kernel that executes on the CUDA device
__global__ void node(Node * nodeInfoList, int seed)
{
  /* threadIdx represents the ID of the thread */
  int i,j;
  int tx = threadIdx.x;
  int numberOfNeighborsToLookAt;
  int neighborIndex;

  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed, 0, 0, &state);
  
  /* continues to loop until the number of days the simulation is set to run */
  while(numberOfDays > 0) {

    /* if thread is not active skip thread activity until everyone else is done */
    if(nodeInfoList[tx].isActive == 1)
    {
    	if (nodeInfoList[tx].nodeStatus == UNINFECTED)
    	{
	    	numberOfNeighborsToLookAt = curand(&state) % (nodeInfoList[tx].numberOfNeighbors + 1);

	    	while( numberOfNeighborsToLookAt > 0 )
	    	{
	    		if (nodeInfoList[tx].nodeStatus == UNINFECTED)
	    		{	    			
		    		do{
		    			neighborIndex = curand(&state) % nodeInfoList[tx].numberOfNeighbors;
		    		} while (neighborIndex == -1);

		    		if(nodeInfoList[neighborIndex].nodeStatus == INFECTIOUS || nodeInfoList[neighborIndex].nodeStatus == ASYMPT)
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


// main routine that executes on the host
int main(void)
{
  int i,j;

  Node * hostNodeInfoList = (Node *) malloc(MAX_NUMBER_OF_NODES*(sizeof(Node)));
 // int * hostNeighborIDs = (int *) malloc( (MAX_NUMBER_OF_NODES*MAX_NUMBER_OF_NEIGHBORS)*(sizeof(int)));

  Node * deviceNodeInfoList;
//  int * deviceNeighborIDs;
 
  cudaMalloc( (void **) &deviceNodeInfoList,(MAX_NUMBER_OF_NODES)* sizeof(Node));
 // cudaMalloc( (void **) &deviceNeighborIDs,(MAX_NUMBER_OF_NODES*MAX_NUMBER_OF_NEIGHBORS)* sizeof(int));
 
  /* setting initial amount of nodes to be active */
  for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  {

	if (i < INIT_NUMBER_OF_NODES) {
        hostNodeInfoList[i].isActive = 1;
	}
    else {
        hostNodeInfoList[i].isActive = 0;
    }

    if(i == 0){
 		hostNodeInfoList[i].nodeStatus = LATENT;
 		hostNodeInfoList[i].dayInfected = 0;
    }
 	else{
 	 	hostNodeInfoList[i].nodeStatus = UNINFECTED;
 	 	hostNodeInfoList[i].dayInfected = -1;
 	}

  }

  for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  {

    if( i < INIT_NUMBER_OF_NODES)
      hostNodeInfoList[i].numberOfNeighbors = (rand() % (MAX_NUMBER_OF_NEIGHBORS + 1));
    else 
      hostNodeInfoList[i].numberOfNeighbors = -1;

    for(j = 0; j < MAX_NUMBER_OF_NEIGHBORS; j++)
    {
      if( j < hostNodeInfoList[i].numberOfNeighbors && i < INIT_NUMBER_OF_NODES)
      {
        hostNodeInfoList[i].neighborId[j] = rand() % INIT_NUMBER_OF_NODES;
      } 
       else 
      {
        hostNodeInfoList[i].neighborId[j] = -1;
      }

    } 

  } 

  /* cudaMemcpy */
  cudaMemcpy(deviceNodeInfoList, hostNodeInfoList, (MAX_NUMBER_OF_NODES) * sizeof(Node), cudaMemcpyHostToDevice); 

  //for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  // for(j = 0; j < MAX_NUMBER_OF_NEIGHBORS; j++)
  //   printf("[%d][%d] = %d\n", i, j,neighborIDs[i][j]);

  dim3 DimGrid(1,1,1);
  dim3 DimBlock(MAX_NUMBER_OF_NODES,1,1);

  node<<<DimGrid,DimBlock>>>(deviceNodeInfoList, time(NULL));

  cudaDeviceSynchronize();

  free(hostNodeInfoList);

  cudaFree(deviceNodeInfoList);

  return 0;

}
