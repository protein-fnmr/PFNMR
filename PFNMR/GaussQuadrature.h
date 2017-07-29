//  PFNMR - Estimate FNMR for proteins (to be updated)
//      Copyright(C) 2016 Jonathan Ellis and Bryan Gantt
//
//  This program is free software : you can redistribute it and / or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation, either version 3 of the License.
//
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//      GNU General Public License for more details.
//
//      You should have received a copy of the GNU General Public License
//      along with this program.If not, see <http://www.gnu.org/licenses/>.

#ifndef __GUASSQUADRATURE_H
#define __GUASSQUADRATURE_H

#include <cfloat>
#define DEF_EPS FLT_EPSILON * 10

class GaussQuadrature
{
public:
	enum INT_METHOD {
		UNSET = -1,
		TEN,
		TWENTY
	};

private:
	INT_METHOD mMethod = UNSET;
	bool bAdaptive;
	float fEps;

public:
	GaussQuadrature(bool adaptive = false, float epsilon = DEF_EPS) :
		bAdaptive(adaptive), fEps(epsilon)
	{
	}
	~GaussQuadrature() { };

	float Intergrate(float begin, float end, float(*func)(const float&));

	void SetMethod(INT_METHOD method) { mMethod = method; };
	INT_METHOD GetMethod() { return mMethod; };
};

#endif
