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

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <locale>
#include <tuple>

#include "PDBProcessor.h"
#include "GPUTypes.h"

using namespace std;

// trimming stuff for when we read in a file
string & ltrim(string & str)
{
    auto it2 = find_if(str.begin(), str.end(), [](char ch) { return !isspace<char>(ch, locale::classic()); });
    str.erase(str.begin(), it2);
    return str;
}

string & rtrim(string & str)
{
    auto it1 = find_if(str.rbegin(), str.rend(), [](char ch) { return !isspace<char>(ch, locale::classic()); });
    str.erase(it1.base(), str.end());
    return str;
}

string & trim(string & str)
{
    return ltrim(rtrim(str));
}

PDBProcessor::PDBProcessor()
{
    cout << "PDBProcessor loaded." << endl;
}

//TODO: This will likely be unnecessary down the road, but is a good general function to have handy just in case.
int PDBProcessor::getAtomsFromPDB(string pdbpath, vector<Atom> & atoms)
{
    ifstream inPdbFile(pdbpath);
    // check if we could even open the file
    if (inPdbFile.is_open())
    {
        string line;

        // read each line
        while (getline(inPdbFile, line))
        {
            // read the first 4 characters
            auto begin = line.substr(0, 4);
            if (begin == "ATOM" || begin == "HETA")
            {
                // make an atom and get all the stuff for it
                Atom curAtom;

                // check the element first to see if we
                // need to keep going or not
                auto element = trim(line.substr(76, 2));

                // default vdw is -1.0f, so only check if we need to change it
                // if it's not in the list, just break out (saves a lot of time)
                // TODO: Perhaps read a property file could be read so user-defined vdws could be used, in case we miss some for things such as metaloenzymes.
                if (element == "H")
                    curAtom.vdw = 1.2f;
                else if (element == "ZN")
                    curAtom.vdw = 1.39f;
                else if (element == "F")
                    curAtom.vdw = 1.47f;
                else if (element == "O")
                    curAtom.vdw = 1.52f;
                else if (element == "N")
                    curAtom.vdw = 1.55f;
                else if (element == "C")
                    curAtom.vdw = 1.7f;
                else if (element == "S")
                    curAtom.vdw = 1.8f;
                else
                    continue;

                curAtom.element = element;

                auto name = line.substr(12, 4);
                auto resName = line.substr(17, 3);
                auto charge = line.substr(78, 2);

                // TODO: Slim this down to the bare essentials, since a lot of this information is not needed.
                // Keep it handy somewhere though, since this is useful reference for reading PDBs.
                curAtom.serial = stoi(line.substr(6, 5));
                curAtom.name = trim(name);
                curAtom.altLoc = line.at(16);
                curAtom.resName = trim(resName);
                curAtom.chainID = line.at(21);
                curAtom.resSeq = stoi(line.substr(22, 4));
                curAtom.iCode = line.at(26);
                curAtom.x = stof(line.substr(30, 8));
                curAtom.y = stof(line.substr(38, 8));
                curAtom.z = stof(line.substr(46, 8));
                curAtom.occupancy = stof(line.substr(54, 6));
                curAtom.tempFactor = stof(line.substr(60, 6));
                curAtom.charge = trim(charge);

                // if we have a valid vdw, add it to the vector
                if (curAtom.vdw != -1.0f)
                    atoms.push_back(curAtom);
            }
        }

        cout << "Found " << atoms.size() << " atoms." << endl;

        // close the file
        inPdbFile.close();
    }
    else
    {
        cout << "Unable to open " << pdbpath << ".  Exiting..." << endl;
        return 1;
    }
}
int PDBProcessor::getGPUAtomsFromPDB(string pdbpath, vector<GPUAtom> & gpuatoms)
{
    ifstream inPdbFile(pdbpath);
    // check if we could even open the file
    if (inPdbFile.is_open())
    {
        string line;

        // read each line
        while (getline(inPdbFile, line))
        {
            // read the first 4 characters
            auto begin = line.substr(0, 4);
            if (begin == "ATOM" || begin == "HETA")
            {
                // make an atom and get all the stuff for it
                GPUAtom curAtom;

                // check the element first to see if we
                // need to keep going or not
                auto element = trim(line.substr(76, 2));

                // default vdw is -1.0f, so only check if we need to change it
                // if it's not in the list, just break out (saves a lot of time)
                // TODO: Perhaps read a property file could be read so user-defined vdws could be used, in case we miss some for things such as metaloenzymes.
                if (element == "H")
                    curAtom.vdw = 1.2f;
                else if (element == "ZN")
                    curAtom.vdw = 1.39f;
                else if (element == "F")
                    curAtom.vdw = 1.47f;
                else if (element == "O")
                    curAtom.vdw = 1.52f;
                else if (element == "N")
                    curAtom.vdw = 1.55f;
                else if (element == "C")
                    curAtom.vdw = 1.7f;
                else if (element == "S")
                    curAtom.vdw = 1.8f;
                else
                    continue;

                curAtom.x = stof(line.substr(30, 8));
                curAtom.y = stof(line.substr(38, 8));
                curAtom.z = stof(line.substr(46, 8));

                // if we have a valid vdw, add it to the vector
                if (curAtom.vdw != -1.0f)
                    gpuatoms.push_back(curAtom);
            }
        }
        auto nAtoms = gpuatoms.size();

        cout << "Found " << nAtoms << " atoms." << endl;
        // close the file
        inPdbFile.close();
        return 0;
    }
    else
    {
        cout << "Unable to open " << pdbpath << ".  Exiting..." << endl;
        return 1;
    }
}