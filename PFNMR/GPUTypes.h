//  PFNMR - Estimate FNMR for proteins (to be updated)
//      Copyright(C) 2016 Jonathan Ellis and Bryan Gantt
//
//  This program is free software : you can redistribute it and / or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation, either version 3 of the License, or
//      (at your option) any later version.
//
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//      GNU General Public License for more details.
//
//      You should have received a copy of the GNU General Public License
//      along with this program.If not, see <http://www.gnu.org/licenses/>.

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