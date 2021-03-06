
#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "adjList.h"
#define NUMBER_OF_NODES 700
#define MAX_NUMBER_OF_NODES 8192
#define MINIMUM_NUMBER_OF_NEIGHBORS 5
#define TOTAL_DAYS_RUN 30

typedef struct
{
	int id;
	int val;
	int numberOfNeighbors;

	int timeInf;
	int infected;
	int diseased;

	NeighborNode * neighborList;
	struct Node * next;
	struct Node * prev;

} Node;

Node * nodeList;

int numberOfNodes;

int adjacencyMatrix[NUMBER_OF_NODES][NUMBER_OF_NODES];



#endif
