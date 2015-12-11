#include "globals.h"

__device__ int currentNumberOfNodes = INIT_NUMBER_OF_NODES;

__device__ int numRem[NUMBER_OF_DAYS];
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
  int tx = threadIdx.x + blockDim.x * blockIdx.x;
  int numberOfNeighborsToLookAt;
  int neighborIndex, index;
  int numberOfDays = 0;
  
  int storedNodeStatus[MAX_NUMBER_OF_NODES];
  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed+(tx*34), 0, 0, &state);
  
  /* continues to loop until the number of days the simulation is set to run */
  while(numberOfDays < NUMBER_OF_DAYS) {

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
	  					neighborIndex = curand(&state) % currentNumberOfNodes;

	  					if(nodeInfoList[tx].neighborId[neighborIndex] == -1)
	  						neighborIndex = -1;
	      		
	      			} while (neighborIndex == -1);

		    		if(storedNodeStatus[neighborIndex] == INFECTIOUS || storedNodeStatus[neighborIndex] == ASYMPT)
			    	{
			    		if(curand(&state) % 100 < 90)
			    		{
			    			nodeInfoList[tx].nodeStatus = LATENT;
			    			nodeInfoList[tx].dayInfected = numberOfDays;
			    		} 

			    		if(curand(&state) % 100 < 10)
			    		{
			    			nodeInfoList[tx].nodeStatus = INCUBATION;
			    			nodeInfoList[tx].dayInfected = numberOfDays;
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
    
		// a chance for node deletion
	    if ( (curand(&state) % 800) < 2 )
	    {
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
	
	if(tx == 0) {
		
		numRem[numberOfDays] = currentNumberOfNodes;
	}
	
	if(nodeInfoList[tx].isActive == 1) {
	
		switch(nodeInfoList[tx].nodeStatus)
		{
			case UNINFECTED:
				atomicAdd(&numUnInf[numberOfDays],1);
			break;
			case LATENT:
				atomicAdd(&numLat[numberOfDays],1);
			break;
			case INCUBATION:
				atomicAdd(&numInc[numberOfDays],1);
			break;
			case INFECTIOUS:
				atomicAdd(&numInf[numberOfDays],1);
			break;
			case ASYMPT:
				atomicAdd(&numAsym[numberOfDays],1);
			break;
			case RECOVERED:
				atomicAdd(&numRec[numberOfDays],1);
			break;
			default:
			
			break;
		}
	
	}
  	
 	numberOfDays++;

    if(nodeInfoList[tx].isActive == 1)
    {

		switch(nodeInfoList[tx].nodeStatus)
		{	
			case UNINFECTED:

			break;
			case LATENT:

				if((numberOfDays) - nodeInfoList[tx].dayInfected >= 2)
				{
					nodeInfoList[tx].nodeStatus = INFECTIOUS;	
				}

			break;
			case INCUBATION:

				if((numberOfDays) - nodeInfoList[tx].dayInfected >= 1)
				{
					nodeInfoList[tx].nodeStatus = ASYMPT;	
				}

			break;
			case INFECTIOUS:

				if((numberOfDays) - nodeInfoList[tx].dayInfected >= 5)
				{
					if( curand(&state) % 100 < (((NUMBER_OF_DAYS - numberOfDays) - nodeInfoList[tx].dayInfected)-5)*10 + 70)
						nodeInfoList[tx].nodeStatus = RECOVERED;	
				}

			break;
			case ASYMPT:

				if((numberOfDays) - nodeInfoList[tx].dayInfected >= 3)
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
	
	if(nodeInfoList[tx].isActive == 0) {
	
		// a chance for node addition 
		if ( ( (float) (curand(&state) % 600) < 2.0 ) )
		{
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

	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	curandState_t state;
	int j, index;

  	curand_init(seed+(tx*56), 0, 0, &state);

  	if(tx < INIT_NUMBER_OF_NODES) {
  		nodeInfoList[tx].isActive = 1;
  		nodeInfoList[tx].numberOfNeighbors = (curand(&state) % (MAX_NUMBER_OF_NEIGHBORS));
  		//printf("Index : %d, number %d\n", tx, nodeInfoList[tx].numberOfNeighbors);

  		for(j = 0; j < MAX_NUMBER_OF_NODES; j++)
  			nodeInfoList[tx].neighborId[j] = -1;

  		for(j = 0; j < nodeInfoList[tx].numberOfNeighbors; j++)
	    {
	    	//if( j < nodeInfoList[tx].numberOfNeighbors) {
		  	
		  		do {
		  			index = curand(&state) % INIT_NUMBER_OF_NODES;

		  			if(nodeInfoList[tx].neighborId[index] != -1)
		  				index = -1;
			  		
			  	} while (index == -1);

			  	nodeInfoList[tx].neighborId[index] = 1;
	      	
	      	//}
	    }
	    
	    //	printf("Index : %d, number %d\n", tx, nodeInfoList[tx].numberOfNeighbors);
 

  	} else {
  		nodeInfoList[tx].isActive = 0;
  		nodeInfoList[tx].numberOfNeighbors = -1;
  	}

  	if(tx == 0)
  	{
  		nodeInfoList[tx].nodeStatus = LATENT;
  		nodeInfoList[tx].dayInfected = 0;

  		for(j = 0; j < NUMBER_OF_DAYS; j++) {
	    	numRem[j] = 0;
	  		numUnInf[j] = 0;
	  	 	numLat[j] = 0;
	  	 	numInf[j] = 0;
	  	 	numInc[j] = 0;
	  	 	numAsym[j] = 0;
	  	 	numRec[j] = 0;
		}

  	} else {

  		nodeInfoList[tx].nodeStatus = UNINFECTED;
  		nodeInfoList[tx].dayInfected = -1;

  	}


}

__global__ void printingRes() 
{
	int numberOfDays = 0;
	int nodeNumber;

	if(threadIdx.x == 0)
	{

		while(numberOfDays < 30) {
		
			nodeNumber = numRem[numberOfDays];
		
			numUnInf[numberOfDays] /= nodeNumber;
	  		numLat[numberOfDays] /= nodeNumber;
	 	 	numInf[numberOfDays] /= nodeNumber;
	 	 	numInc[numberOfDays] /= nodeNumber;
		  	numAsym[numberOfDays] /= nodeNumber;
		  	numRec[numberOfDays] /= nodeNumber;
	  	
		  	numUnInf[numberOfDays] *= 100;
	  		numLat[numberOfDays] *= 100;
		  	numInf[numberOfDays] *= 100;
		  	numInc[numberOfDays] *= 100;
		  	numAsym[numberOfDays] *= 100;
		  	numRec[numberOfDays] *= 100;

			printf("\n \nDay %d Number of Nodes: %d\n",numberOfDays,numRem[numberOfDays]);

			printf("Percent Uninfected: %f, Num Latent %f, Num Inf %f, Num Inc %f, Num Asym %f, Num Rec %f\n", numUnInf[numberOfDays],
			 numLat[numberOfDays], numInf[numberOfDays], numInc[numberOfDays], numAsym[numberOfDays],
			  numRec[numberOfDays]);
			  
			numberOfDays++;

		}

	}

}

// main routine that executes on the host
int main(void)
{
  Node * hostNodeInfoList = (Node *) malloc(MAX_NUMBER_OF_NODES*(sizeof(Node)));

  Node * deviceNodeInfoList;
 
  cudaMalloc( (void **) &deviceNodeInfoList,(MAX_NUMBER_OF_NODES)* sizeof(Node));
 
  cudaMemcpy(deviceNodeInfoList, hostNodeInfoList, (MAX_NUMBER_OF_NODES) * sizeof(Node), cudaMemcpyHostToDevice); 

  dim3 DimGrid( (int) ceil(MAX_NUMBER_OF_NODES/512.0),1,1);
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

