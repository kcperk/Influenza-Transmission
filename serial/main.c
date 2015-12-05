
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "graph.h"
#include "stack.c"

void initGraph();
void DFSinfection();
void printCurrentDiseaseStatus(int currentTimeStamp);
int updateSurvivors(int currentTimeStamp);
void printGraph();
int addNode();
void removeNode(int idToRemove);
Node * findNode(int index);

int main() {

	srand(time(NULL));

	initGraph();

	DFSinfection();

	//printGraph();


	//removeNode(0);

	//printGraph();

	printf("\n");

	//addNode();

//	printGraph();

	return 0;
}

void DFSinfection() {

	int guess1, guess2, guess3;
	int i;
	int index;
	int check;
	int currentTimeStamp = 0;
	Stack * nodeStack = (Stack *) malloc(sizeof(Stack));
	Stack * nextStack = (Stack *) malloc(sizeof(Stack));
	Node * currentNode;
	NeighborNode * currentNeighbor;

	Stack_Init(nodeStack);

	Stack_Push(nodeStack,0);

	while (currentTimeStamp < TOTAL_DAYS_RUN)
	{
		 index = Stack_Pop(nodeStack);

		 currentNode = findNode(index);

		 if(currentNode != NULL) 
		 {

			 if (currentNode->infected == 0)
			 {

			 	guess3 = rand() % 100;

			 	if (guess3 < 50) {

				 	currentNode->infected = 1;
				 	currentNode->timeInf = currentTimeStamp;

				 	guess1 = rand() % 100;
				 	guess2 = rand() % 70;

				 	if ( guess1 < guess2)
				 	{
				 		currentNode->diseased = 1;
				 	}

				 	currentNeighbor = currentNode->neighborList;

				 	while(currentNeighbor != NULL) {

				 		Stack_Push(nextStack,currentNeighbor->id);
				 		currentNeighbor = currentNeighbor->next;

				 	}

			 	}

			 }

		}

		if(nodeStack->topOfStack == 0)
		{
			printCurrentDiseaseStatus(currentTimeStamp);
			check = updateSurvivors(currentTimeStamp);
			currentTimeStamp++;

			free(nodeStack);
			nodeStack = nextStack;

			if(check != -1)
				Stack_Push(nodeStack, check);

			nextStack = (Stack *) malloc(sizeof(Stack));

		}

    
	}

}

int addNode(){

	Node * newNode;
	
	Node * prevNode;

	Node * testNode;

	Node * otherNode;

	NeighborNode * otherNeighbor;

	NeighborNode * otherNewNeighbor;

	NeighborNode * newNeighbor; // Used to create new neighbors
	
	NeighborNode * currentNeighbor; // Used to traverse neighbor lists

	NeighborNode * tempNeighbor; // Used to set prev node in doubly linked list

	NeighborNode * testNeighbor; //Used to test if neighbor has not already been added

	int uniqueNumBool, newID, nodeToAdd, numberOfNeighborsToBeAdded;
	int res = -1;

	if(numberOfNodes == MAX_NUMBER_OF_NODES)
		return;

	newNode = (Node *) malloc(sizeof(Node));
	prevNode = nodeList;

	/* Generating a new ID for a node */
	uniqueNumBool = 0;
	while(!uniqueNumBool) {

		uniqueNumBool = 1;
		/* Random Node ID */
		newID = rand() % MAX_NUMBER_OF_NODES;

		testNode = nodeList;
		while(testNode != NULL)
		{
			if(testNode->id == newID)
			{
				uniqueNumBool = 0;
				break;
			}
			
			testNode = testNode->next;
		}

	}

	newNode->id = newID;
	newNode->numberOfNeighbors = 0;
	newNode->infected = 0;
	newNode->diseased = 0;

	while(prevNode->next != NULL)
		prevNode = prevNode->next;

	/* Adding the newNode to the nodeList */
	prevNode->next = newNode;

	numberOfNeighborsToBeAdded = rand() % (numberOfNodes);

	if (numberOfNeighborsToBeAdded < MINIMUM_NUMBER_OF_NEIGHBORS)
		numberOfNeighborsToBeAdded = MINIMUM_NUMBER_OF_NEIGHBORS; 

	numberOfNeighborsToBeAdded -= newNode->numberOfNeighbors;

	while(numberOfNeighborsToBeAdded > 0) 
	{

		uniqueNumBool = 0;
		while(!uniqueNumBool) {

			uniqueNumBool = 1;
			/* Random Node Neighbor */
			nodeToAdd = rand() % numberOfNodes;

			testNeighbor = newNode->neighborList;
			while(testNeighbor != NULL)
			{
				if(testNeighbor->id == nodeToAdd)
				{
					uniqueNumBool = 0;
					break;
				}
				testNeighbor = testNeighbor->next;
			}

			 /*neighbor can't be same as current node */
			if(nodeToAdd == newNode->id)
				uniqueNumBool = 0;

			if(findNode(nodeToAdd) == NULL)
				uniqueNumBool = 0;

		}

		newNeighbor = (NeighborNode *) malloc(sizeof(NeighborNode));
		newNeighbor->id = nodeToAdd;

		otherNewNeighbor = (NeighborNode *) malloc(sizeof(NeighborNode));
		otherNewNeighbor->id = newNode->id;

		otherNode = findNode(nodeToAdd);
		res = nodeToAdd;

		currentNeighbor = newNode->neighborList;
		otherNeighbor = otherNode->neighborList;

		if(currentNeighbor == NULL) {
			newNode->neighborList = newNeighbor;
			newNode->numberOfNeighbors++;
		}
		else {

			while(currentNeighbor->next != NULL)
				currentNeighbor = currentNeighbor->next;

			currentNeighbor->next = newNeighbor;
			newNeighbor->prev = currentNeighbor;
			newNode->numberOfNeighbors++;

		}

		if(otherNeighbor == NULL){
			otherNode->neighborList = otherNewNeighbor;
			otherNode->numberOfNeighbors++;
		} else {

			while(otherNeighbor->next != NULL)
				otherNeighbor = otherNeighbor->next;

			otherNeighbor->next = otherNewNeighbor;
			otherNewNeighbor->prev = otherNeighbor;
			otherNode->numberOfNeighbors++;
		}

		numberOfNeighborsToBeAdded--;
	}

	numberOfNodes++;

	return res;
}


void removeNode(int idToRemove)
{
	Node * targetNode = findNode(idToRemove);
	NeighborNode * currentNeighbor = targetNode->neighborList;

	Node * checkingNode;
	NeighborNode * removingNeighbor;

	Node * prevNode = nodeList;

	NeighborNode * previousNeigh;
	NeighborNode * nextNeigh;

	/* Removing node from all other nodes neighbor's list */
	while(currentNeighbor != NULL)
	{
		checkingNode = findNode(currentNeighbor->id);
		removingNeighbor = checkingNode->neighborList;

		while(removingNeighbor != NULL)
		{

			if(removingNeighbor->id == idToRemove)
			{

				if(removingNeighbor->prev != NULL && removingNeighbor->next != NULL)
				{
					previousNeigh = removingNeighbor->prev;
					nextNeigh = removingNeighbor->next;
					previousNeigh->next = nextNeigh;
					nextNeigh->prev = previousNeigh;
				}
				else if(removingNeighbor->prev == NULL && removingNeighbor->next != NULL)
				{
					nextNeigh = removingNeighbor->next;
					nextNeigh->prev = NULL;
					checkingNode->neighborList = nextNeigh;
				} 
				else if(removingNeighbor->prev != NULL && removingNeighbor->next == NULL)
				{
					previousNeigh = removingNeighbor->prev;
					previousNeigh->next = NULL;
				}	

				free(removingNeighbor);

			}

			removingNeighbor = removingNeighbor->next;
		}

		currentNeighbor = currentNeighbor->next;
	}

	/* Removing node */
	if(prevNode == targetNode)
	{
		if(targetNode->next != NULL)
			nodeList = targetNode->next;
		else 
			nodeList = NULL;
	
	} else {

		while(prevNode->next != targetNode)
			prevNode = prevNode->next;

		if(targetNode->next != NULL)
			prevNode->next = targetNode->next;
		else 
			prevNode->next = NULL;


	}	

	numberOfNodes--;

	free(targetNode);

}


int updateSurvivors(int currentTimeStamp)
{

	Node * currentNode = nodeList;
	int chanceToBeCured;
	int chanceToDie;
	int chanceForBirth;
	int res = -1;

	while(currentNode != NULL) {

		if(currentNode->diseased)
		{

			chanceToBeCured = rand() % 100;
			chanceToDie = rand() % 100;

			if(chanceToBeCured < (currentTimeStamp - currentNode->timeInf)*10 )
				currentNode->diseased = 0;

			if(chanceToDie < 10)
			{
				removeNode(currentNode->id);
				currentNode = currentNode->next;
				continue;
			}

		}

		currentNode = currentNode->next;
	}

	chanceForBirth = rand() % 100;

	if(chanceForBirth < 5)
		res = addNode();

	return res;

}


void printCurrentDiseaseStatus(int currentTimeStamp)
{

	Node * currentNode = nodeList;
	float numInf = 0;
	float numDis = 0;
	float numFree = 0;

	float infTotal = 0;
	float disTotal = 0;
	float freTotal = 0;


//	printf("\nDay %d \nNumber of Infected Nodes:", currentTimeStamp);
	while(currentNode != NULL)
	{

		if(currentNode->infected && !currentNode->diseased)
		{
	//		printf(" Node %d", currentNode->id);
			numInf++;

		}

		currentNode = currentNode->next;
	}

	currentNode = nodeList;
	//printf("\nNumber of Diseased Nodes:");

	while(currentNode != NULL)
	{

		if(currentNode->diseased)
		{
	//		printf(" Node %d", currentNode->id);
			numDis++;

		}
		
		currentNode = currentNode->next;
	}

	currentNode = nodeList;
	//printf("\nNumber of Disease Free Nodes:");

	while(currentNode != NULL)
	{

		if(!currentNode->infected)
		{
	//		printf(" Node %d", currentNode->id);
			numFree++;

		}
		
		currentNode = currentNode->next;
	}

	infTotal = (numInf/numberOfNodes) * 100;
	disTotal = (numDis/numberOfNodes) * 100;
	freTotal = (numFree/numberOfNodes) * 100;

	printf("\nDay %d Number Of People: %d Total Percentages:\n%f Carriers | %f Diseased | %f Free\n", currentTimeStamp, numberOfNodes, infTotal, disTotal, freTotal);
}

/* Make sure that index exists */
Node * findNode(int index) {

	Node * currentNode;

	currentNode = nodeList;

	while (currentNode != NULL) {

		if(currentNode->id == index)
			return currentNode;

		currentNode = currentNode->next;
	}

	currentNode = NULL;
	return currentNode;

}



void initGraph() {

	int i,j;
	int numberOfNeighborsToBeAdded, uniqueNumBool, nodeToAdd;
	
	Node * newNode; // Used to create a new graph node

	Node * currentNode; // Used to traverse node list

	Node * otherNode;

	NeighborNode * otherNeighbor;

	NeighborNode * otherNewNeighbor;

	NeighborNode * newNeighbor; // Used to create new neighbors
	
	NeighborNode * currentNeighbor; // Used to traverse neighbor lists

	NeighborNode * tempNeighbor; // Used to set prev node in doubly linked list

	NeighborNode * testNeighbor; //Used to test if neighbor has not already been added

	/* Creating Nodes */

	newNode = (Node *) malloc(sizeof(Node));

	newNode->id = 0;
	newNode->numberOfNeighbors = 0;
	newNode->infected = 0;
	newNode->diseased = 0;

	nodeList = newNode;
	currentNode = nodeList;

	for(i = 1; i < NUMBER_OF_NODES; i++)
	{
		newNode = (Node *) malloc(sizeof(Node));

		newNode->id = i;
		newNode->numberOfNeighbors = 0;
		newNode->infected = 0;
		newNode->diseased = 0;
		
		currentNode->next = newNode;
		currentNode = newNode;

	}

	numberOfNodes = NUMBER_OF_NODES;

	/* Adding Edges */
	currentNode = nodeList;

	while(currentNode != NULL) {

		numberOfNeighborsToBeAdded = rand() % (NUMBER_OF_NODES);

		if (numberOfNeighborsToBeAdded < MINIMUM_NUMBER_OF_NEIGHBORS)
			numberOfNeighborsToBeAdded = MINIMUM_NUMBER_OF_NEIGHBORS; 

		numberOfNeighborsToBeAdded -= currentNode->numberOfNeighbors;

		while(numberOfNeighborsToBeAdded > 0) {

			uniqueNumBool = 0;
			while(!uniqueNumBool) {

				uniqueNumBool = 1;
				/* Random Node Neighbor */
				nodeToAdd = rand() % NUMBER_OF_NODES;

				testNeighbor = currentNode->neighborList;
				while(testNeighbor != NULL)
				{
					if(testNeighbor->id == nodeToAdd)
					{
						uniqueNumBool = 0;
						break;
					}
					testNeighbor = testNeighbor->next;
				}

				 /*neighbor can't be same as current node */
				if(nodeToAdd == currentNode->id)
					uniqueNumBool = 0;

			}

			newNeighbor = (NeighborNode *) malloc(sizeof(NeighborNode));
			newNeighbor->id = nodeToAdd;

			otherNewNeighbor = (NeighborNode *) malloc(sizeof(NeighborNode));
			otherNewNeighbor->id = currentNode->id;

			otherNode = findNode(nodeToAdd);

			currentNeighbor = currentNode->neighborList;
			otherNeighbor = otherNode->neighborList;

			if(currentNeighbor == NULL) {
				currentNode->neighborList = newNeighbor;
				currentNode->numberOfNeighbors++;
			}
			else {

				while(currentNeighbor->next != NULL)
					currentNeighbor = currentNeighbor->next;

				currentNeighbor->next = newNeighbor;
				newNeighbor->prev = currentNeighbor;
				currentNode->numberOfNeighbors++;

			}

			if(otherNeighbor == NULL){
				otherNode->neighborList = otherNewNeighbor;
				otherNode->numberOfNeighbors++;
			} else {

				while(otherNeighbor->next != NULL)
					otherNeighbor = otherNeighbor->next;

				otherNeighbor->next = otherNewNeighbor;
				otherNewNeighbor->prev = otherNeighbor;
				otherNode->numberOfNeighbors++;
			}

			numberOfNeighborsToBeAdded--;
		}

		currentNode = currentNode->next;
	}


}

void printGraph() {

	int i,j;
	Node * currentNode = nodeList;
	NeighborNode * currentNeighbor;

	if (currentNode == NULL)
	{
		printf("Graph is not valid");
		return;

	}

	while(currentNode != NULL) {

		printf("\n-------------------\nNode Number: %d \n", currentNode->id);

		if(currentNode->diseased && currentNode->infected)
			printf("This person is diseased\n");

		if(!currentNode->diseased && currentNode->infected)
			printf("This person is a carrier\n");

		if(currentNode->diseased && !currentNode->infected)
			printf("This is impossible, something went wrong\n");

		if(!currentNode->diseased && !currentNode->infected)
			printf("This person is disease free!!!!\n");

		currentNeighbor = currentNode->neighborList;
		while(currentNeighbor != NULL) {

			printf("Node %d\n", currentNeighbor->id);

			currentNeighbor = currentNeighbor->next;
		}

		currentNode = currentNode->next;

	}

}



