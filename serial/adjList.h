#ifndef _ADJLIST_H_
#define _ADJLIST_H_

typedef struct
{
	int id;

	struct NeighborNode * next;
	struct NeighborNode * prev;

} NeighborNode;

#endif

