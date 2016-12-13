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

#ifndef __CALCULATIONMETHODS_H
#define __CALCULATIONMETHODS_H

#include <vector>

void getGaussQuadSetup(int points, vector<float> & outWeights, vector<float> & outAbscissa);
int oldElectricFieldCalculation(string pdbPath, const float lineresolution, const float inDielectric, const float outDielectric, const float variance);
int electricFieldCalculation(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance);
int electricPotentialCalculation(string pdbPath, const int integralres, const int nSlices, const int gridres, const float inDielectric, const float outDielectric, const float variance);

#endif