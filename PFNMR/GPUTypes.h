// More than just GPU types, but still
// structs are good because we don't need any class nonsense

#ifndef __GPUTYPES_H
#define __GPUTYPES_H

#include <string>

using namespace std;

typedef struct {
	//int resID;
	float x;
	float y;
	float z;
	float vdw;
} GPUAtom;

typedef struct {
	int serial;
	string name;
	char altLoc;
	string resName;
	char chainID;
	int resSeq;
	char iCode;
	float x;
	float y;
	float z;
	float occupancy;
	float tempFactor;
	string element;
	string charge;
	float vdw;
	float density;
	float dielectric;
} Atom;

typedef struct {
	float x;
	float y;
	float z;
	float dielectric = -1.0;
} GridPoint;

#endif // !__GPUTYPES_H