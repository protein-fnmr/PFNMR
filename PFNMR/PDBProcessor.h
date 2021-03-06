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

// C++ code for reading and processing PDB files

#ifndef PDBPROCESSOR_H
#define PDBPROCESSOR_H

#include <string>
#include <array>
#include <vector>

#include "GPUTypes.h"

using namespace std;

class PDBProcessor {

private:
    string pdbPath;
    ifstream pdbStream;
    bool isOpen = false;

public:
    PDBProcessor(string pdbPath);
    ~PDBProcessor();

    bool is_open() { return isOpen; }

    vector<GPUAtom> getGPUAtoms();
    vector<GPUChargeAtom> getGPUChargeAtoms(vector<vector<string>> & chargetable);
    vector<Atom> PDBProcessor::getAtomsFromPDB();
    vector<GPUChargeAtom> getGPUChargeAtomsFromAtoms(vector<Atom> & atoms, vector<vector<string>> & chargetable);

};

#endif