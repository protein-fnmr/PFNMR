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

vector<float> crossprod(vector<float> & a, vector<float> & b);
void getGaussQuadSetup(int points, vector<float> & outWeights, vector<float> & outAbscissa);
vector<float> rotateResidueToXField(vector<float> & fieldVect, vector<Atom> & residue);
vector<float> generateGeometryRotationAnglesToX(vector<float> & geomVec);
//int createDielectricPFDFile(string outpfdpath, string pdbFilePath, string colorcsvpath, int nSlices, int imgSize, float outDielectric, float inDielectric, float relVariance);
//int oldElectricFieldCalculation(string pdbPath, const float lineresolution, const float inDielectric, const float outDielectric, const float variance);
int electricFieldCalculation(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance, vector<float> & output);
int electricPotentialCalculation(string pdbPath, const int integralres, const int nSlices, const int gridres, const float inDielectric, const float outDielectric, const float variance);
float calculateAverageDielectric(int numpoints, float sphererad, vector<GPUAtom> atoms, GPUAtom & target, float variance, float inDielectric, float outDielectric);
float pheNMR(float x, float y, float z, float d, float w);

#endif