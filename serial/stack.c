
#include "graph.h"

typedef struct
{
    int    data[NUMBER_OF_NODES];
    int     topOfStack;

} Stack;


void Stack_Init(Stack *S)
{
    S->topOfStack = 0;
}

void Stack_Push(Stack *S, int d)
{
    if(S->topOfStack == NUMBER_OF_NODES)
    {    
        //printf("Error: stack full\n");
        return;
    }

    S->data[S->topOfStack] = d;
    S->topOfStack++;    
}

int Stack_Pop(Stack *S)
{
   int res;

    if (S->topOfStack == 0) {
        //printf("Error: stack empty\n");
        return;
    } 

    S->topOfStack--;
    res = S->data[S->topOfStack];
    return res;
}

