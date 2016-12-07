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

// C++ code for writing PFD files

#ifndef PFDPROCESSOR_H
#define PFDPROCESSOR_H

#include <vector>
#include <string>

#include <glm/glm.hpp>

#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include "GPUTypes.h"

struct PFDWriter
{
    ofstream file;
};

struct PFDReader
{
    ifstream file;
};

bool openPFDFileWriter(PFDWriter* writer, string outpath);
bool openPFDFileReader(PFDReader* reader, string inpath);
void closePFDFileWriter(PFDWriter* writer);
void closePFDFileReader(PFDReader* reader);
void writeStructurePFDInfo(PFDWriter* writer, vector<Atom> & atoms, vector<vector<string>> & colorcsv);
void writeDielectricFrameData(PFDWriter* writer, const uint8_t* image, vector<float> & planeDims, uint32_t imgSideResolution);
bool loadPFDFile(PFDReader* reader, std::vector<glm::vec3> & out_atomverts, std::vector<glm::vec3> & out_atomcols, std::vector<unsigned short> & out_bondindicies);
bool loadPFDTextureFile(PFDReader* reader, vector<glm::vec3> & out_atomverts, vector<glm::vec4> & out_atomcols, vector<unsigned short> & out_bondindicies, vector<glm::vec3> & out_texverts, vector<GLuint> & texIDs);
#endif