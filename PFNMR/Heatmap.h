// C++ implementation of the heatmap code
// it's fairly self-explanitory if you know the C# code
// it's like 90% the same

#ifndef HEATMAP_H
#define HEATMAP_H

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

#endif // HEATMAP_H