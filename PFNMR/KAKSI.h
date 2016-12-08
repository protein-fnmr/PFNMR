#pragma once
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

// C++ code for implementing KAKSI method for secondary structure determination
// Method taken from: "Protein secondary structure assignment revisited: a detailed analysis of different assignment methods", 
//                           Juliette Martin et. al, BMC Structural Biology, doi:10.1186/1472-6807-5-17
#ifndef KAKSI_H
#define KAKSI_H

#include <vector>

#include "GPUTypes.h"

bool determineSecondaryStructureCPU(vector<Atom> & atoms, vector<vector<Atom>> & out_helicies, vector<vector<Atom>> & out_sheets);

#endif