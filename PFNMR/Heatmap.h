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

// C++ implementation of the heatmap code

#ifndef __HEATMAP_H
#define __HEATMAP_H

#include <math.h>

#define NUM_COLORS 5

float* getHeatMapColor(float value)
{
    float color[NUM_COLORS][3] = { { 0.0f, 0.0f, 1.0f },{ 0.0f, 1.0f, 1.0f },{ 0.0f, 1.0f, 0.0f },{ 1.0f, 1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } };
    int idx1 = 0;
    int idx2 = 0;
    float fractBetween = 0.0f;

    if (value > 0)
    {
        if (value >= 1)
        {
            idx1 = NUM_COLORS - 1;
            idx2 = NUM_COLORS - 2;
        }
        else
        {
            value *= (NUM_COLORS - 1);
            idx1 = (int)floor(value);
            idx2 = idx1 + 1;
            fractBetween = value - idx1;
        }
    }

    float *heatmap = new float[3];
    heatmap[0] = (color[idx2][0] - color[idx1][0]) * fractBetween + color[idx1][0];
    heatmap[1] = (color[idx2][1] - color[idx1][1]) * fractBetween + color[idx1][1];
    heatmap[2] = (color[idx2][2] - color[idx1][2]) * fractBetween + color[idx1][2];

    return heatmap;
}

#endif // !__HEATMAP_H