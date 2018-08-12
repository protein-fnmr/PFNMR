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

#ifndef __TRAININGMETHODS_H
#define __TRAININGMETHODS_H

#include <vector>
int electricFieldCalculationSilent(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance, vector<float> & output);
int electricFieldCalculationSilentReporter(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance, vector<float> & output, vector<float> & outputneg, vector<float> & flipoutput, vector<float> & flipoutputneg, ofstream & report);
int electricFieldCalculationGradientOpt(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance, const float powvar, const float multvar, vector<float> & output, vector<float> & outputneg, vector<float> & flipoutput, vector<float> & flipoutputneg, vector<float> parameters);
void gradientOptFunc(string pdbFilePath, int res, int optparam, float stepsize, float lasterror, vector<vector<float>> & parameters, vector<float> correctshifts, ofstream & logfile);
float errorfunc(float calculated, float correct);
float rounderhelper(float x, float step);

#endif