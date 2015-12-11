#include "globals.h"

__device__ int currentNumberOfNodes = INIT_NUMBER_OF_NODES;
__device__ int numberOfDays = NUMBER_OF_DAYS;

__device__ float numRem[NUMBER_OF_DAYS];
__device__ float numUnInf[NUMBER_OF_DAYS];
__device__ float numLat[NUMBER_OF_DAYS];
__device__ float numInf[NUMBER_OF_DAYS];
__device__ float numInc[NUMBER_OF_DAYS];
__device__ float numAsym[NUMBER_OF_DAYS];
__device__ float numRec[NUMBER_OF_DAYS];


// Kernel that executes on the CUDA device
__global__ void node(Node * nodeInfoList, int seed)
{
  /* threadIdx represents the ID of the thread */
  int i,j;
  int tx = threadIdx.x;
  int numberOfNeighborsToLookAt;
  int neighborIndex, index;
  
  
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
    
    
    if(nodeInfoList[tx].isActive == 1)
    {
    
	/* a chance for node deletion */
    if ( (curand(&state) % 100) < 5 )
    {
    
    	//printf("\nRemoving Node %d",tx);
    
    	nodeInfoList[tx].isActive = 0;
    	nodeInfoList[tx].id = 0;
		nodeInfoList[tx].nodeStatus = UNINFECTED;

    	for(i = 0; i < MAX_NUMBER_OF_NEIGHBORS; i++)
    		nodeInfoList[tx].neighborId[i] = -1;

    	nodeInfoList[tx].numberOfNeighbors = 0;
    	
    	atomicAdd(&currentNumberOfNodes,-1);

    }

	}

	__syncthreads();
	
    if ( tx == 0) {

   		// printf("\n \nDay %d Number of Nodes: %d\n",NUMBER_OF_DAYS - numberOfDays,currentNumberOfNodes);

    	numRem[NUMBER_OF_DAYS - numberOfDays] = currentNumberOfNodes;
  		numUnInf[NUMBER_OF_DAYS - numberOfDays] = 0;
  	 	numLat[NUMBER_OF_DAYS - numberOfDays] = 0;
  	 	numInf[NUMBER_OF_DAYS - numberOfDays] = 0;
  	 	numInc[NUMBER_OF_DAYS - numberOfDays] = 0;
  	 	numAsym[NUMBER_OF_DAYS - numberOfDays] = 0;
  	 	numRec[NUMBER_OF_DAYS - numberOfDays] = 0;

    	for(i = 0; i < MAX_NUMBER_OF_NODES; i++)
  		{	
	  		switch(nodeInfoList[i].nodeStatus)
	  		{
	  			case UNINFECTED:
	  				numUnInf[NUMBER_OF_DAYS - numberOfDays]++;
				break;
				case LATENT:
					numLat[NUMBER_OF_DAYS - numberOfDays]++;
				break;
				case INCUBATION:
					numInc[NUMBER_OF_DAYS - numberOfDays]++;
				break;
				case INFECTIOUS:
					numInf[NUMBER_OF_DAYS - numberOfDays]++;
				break;
				case ASYMPT:
					numAsym[NUMBER_OF_DAYS - numberOfDays]++;
				break;
				case RECOVERED:
					numRec[NUMBER_OF_DAYS - numberOfDays]++;
				break;
				default:
					
				break;
	  		}
  	
  		}
  	
	  	numUnInf[NUMBER_OF_DAYS - numberOfDays] /= MAX_NUMBER_OF_NODES;
	  	numLat[NUMBER_OF_DAYS - numberOfDays] /= MAX_NUMBER_OF_NODES;
	  	numInf[NUMBER_OF_DAYS - numberOfDays] /= MAX_NUMBER_OF_NODES;
	  	numInc[NUMBER_OF_DAYS - numberOfDays] /= MAX_NUMBER_OF_NODES;
	  	numAsym[NUMBER_OF_DAYS - numberOfDays] /= MAX_NUMBER_OF_NODES;
	  	numRec[NUMBER_OF_DAYS - numberOfDays] /= MAX_NUMBER_OF_NODES;
	  	
	  	numUnInf[NUMBER_OF_DAYS - numberOfDays] *= 100;
	  	numLat[NUMBER_OF_DAYS - numberOfDays] *= 100;
	  	numInf[NUMBER_OF_DAYS - numberOfDays] *= 100;
	  	numInc[NUMBER_OF_DAYS - numberOfDays] *= 100;
	  	numAsym[NUMBER_OF_DAYS - numberOfDays] *= 100;
	  	numRec[NUMBER_OF_DAYS - numberOfDays] *= 100;
	  	
	  //	printf("Number Uninfected: %f, Num Latent %f, Num Inf %f, Num Inc %f, Num Asym %f, Num Rec %f\n", numUnInf, numLat, numInf, numInc, numAsym, numRec);
	  	

   }

    __syncthreads();

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
	
	
	if(nodeInfoList[tx].isActive == 0) {
	
		/* a chance for node addition */
		if ( (curand(&state) % 100) < 5 )
		{
			//printf("\nAdding Node %d", tx);
		
			nodeInfoList[tx].isActive = 1;
			nodeInfoList[tx].nodeStatus = UNINFECTED;
	
			nodeInfoList[tx].numberOfNeighbors = (curand(&state) % (MAX_NUMBER_OF_NEIGHBORS + 1));

			for(j = 0; j < MAX_NUMBER_OF_NODES; j++)
	  			nodeInfoList[tx].neighborId[j] = -1;

	  		for(j = 0; j < MAX_NUMBER_OF_NEIGHBORS; j++)
			{
		  		do {
		  			index = curand(&state) % currentNumberOfNodes;

		  			if(nodeInfoList[tx].neighborId[index] != -1)
		  				index = -1;
			  		
			  	} while (index == -1);

			  	nodeInfoList[tx].neighborId[index] = 1;
			}
			
			atomicAdd(&currentNumberOfNodes,1); 

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

__global__ void printingRes() 
{
	int numberOfDays = 30;

	if(threadIdx.x == 0)
	{

		while(numberOfDays > 0) {

			printf("\n \nDay %d Number of Nodes: %d\n",NUMBER_OF_DAYS - numberOfDays,currentNumberOfNodes);

			printf("Number Uninfected: %f, Num Latent %f, Num Inf %f, Num Inc %f, Num Asym %f, Num Rec %f\n", numUnInf[NUMBER_OF_DAYS - numberOfDays], numLat[NUMBER_OF_DAYS - numberOfDays], numInf[NUMBER_OF_DAYS - numberOfDays], numInc[NUMBER_OF_DAYS - numberOfDays], numAsym[NUMBER_OF_DAYS - numberOfDays], numRec[NUMBER_OF_DAYS - numberOfDays]);
	  	
			numberOfDays--;

		}

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

  dim3 DimGrid(ceil(MAX_NUMBER_OF_NODES/512.0),1,1);
  dim3 DimBlock(512,1,1);

  initGraph<<<DimGrid,DimBlock>>>(deviceNodeInfoList, time(NULL));

  cudaDeviceSynchronize();

  node<<<DimGrid,DimBlock>>>(deviceNodeInfoList, time(NULL));

  cudaDeviceSynchronize();

  printingRes<<<1,1>>>();

  cudaDeviceSynchronize();

  free(hostNodeInfoList);

  cudaFree(deviceNodeInfoList);

  return 0;

}

