#ifndef GLOBALS_H
#define GLOBALS_H

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>


#define INIT_NUMBER_OF_NODES 10
#define MAX_NUMBER_OF_NODES 30
#define MIN_NUMBER_OF_NEIGHBORS 1
#define MAX_NUMBER_OF_NEIGHBORS 4

#define MAX_X 200
#define MAX_Y 200

#define NUMBER_OF_DAYS 30

/* Infection Status */

#define UNINFECTED 0

/* Cannot Infect Others 2 Day Period. 90% probablility of infection */
#define LATENT 1 

/* Cannot Infect Others 1 Day Period. 10% */
#define INCUBATION 2

/* Can Infect Others 3-5 Day Period */
#define INFECTIOUS 3

/* Can Infect Others 2 Day Period */
#define ASYMPT 4 

/* Has recovered, Cannot be infected Again*/
#define RECOVERED 5

typedef struct
{
	int	isActive;
	int id;
	int numberOfNeighbors;
	int nodeStatus;

	int dayInfected;
	int neighborId[MAX_NUMBER_OF_NEIGHBORS];

} Node;


#endif

