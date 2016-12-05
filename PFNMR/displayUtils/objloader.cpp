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

// C++ code for doing object file processing
// Made from code taken from http://www.opengl-tutorial.org/, beer is in the mail.
#include <vector>
#include <stdio.h>
#include <string>
#include <cstring>
#include <fstream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "objloader.h"

using namespace std;


// Very, VERY simple OBJ loader.
// Here is a short list of features a real function would provide : 
// - Binary files. Reading a model should be just a few memcpy's away, not parsing a file at runtime. In short : OBJ is not very great.
// - Animations & bones (includes bones weights)
// - Multiple UVs
// - All attributes should be optional, not "forced"
// - More stable. Change a line in the OBJ file and it crashes.
// - More secure. Change another line and you can inject code.
// - Loading from memory, stream, etc

bool loadOBJ(
    const char * path,
    vector<glm::vec3> & out_vertices,
    vector<glm::vec2> & out_uvs,
    vector<glm::vec3> & out_normals
) {
    printf("Loading OBJ file %s...\n", path);

    vector<unsigned int> vertexIndices, uvIndices, normalIndices;
    vector<glm::vec3> temp_vertices;
    vector<glm::vec2> temp_uvs;
    vector<glm::vec3> temp_normals;


    FILE * file = fopen(path, "r");
    if (file == NULL) {
        printf("Impossible to open the file ! Are you in the right path ? See Tutorial 1 for details\n");
        getchar();
        return false;
    }

    while (1) {

        char lineHeader[128];
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

                   // else : parse lineHeader

        if (strcmp(lineHeader, "v") == 0) {
            glm::vec3 vertex;
            fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
            temp_vertices.push_back(vertex);
        }
        else if (strcmp(lineHeader, "vt") == 0) {
            glm::vec2 uv;
            fscanf(file, "%f %f\n", &uv.x, &uv.y);
            uv.y = -uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
            temp_uvs.push_back(uv);
        }
        else if (strcmp(lineHeader, "vn") == 0) {
            glm::vec3 normal;
            fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
            temp_normals.push_back(normal);
        }
        else if (strcmp(lineHeader, "f") == 0) {
            string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
            if (matches != 9) {
                printf("File can't be read by our simple parser :-( Try exporting with other options\n");
                return false;
            }
            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);
            uvIndices.push_back(uvIndex[0]);
            uvIndices.push_back(uvIndex[1]);
            uvIndices.push_back(uvIndex[2]);
            normalIndices.push_back(normalIndex[0]);
            normalIndices.push_back(normalIndex[1]);
            normalIndices.push_back(normalIndex[2]);
        }
        else {
            // Probably a comment, eat up the rest of the line
            char stupidBuffer[1000];
            fgets(stupidBuffer, 1000, file);
        }

    }

    // For each vertex of each triangle
    for (unsigned int i = 0; i<vertexIndices.size(); i++) {

        // Get the indices of its attributes
        unsigned int vertexIndex = vertexIndices[i];
        unsigned int uvIndex = uvIndices[i];
        unsigned int normalIndex = normalIndices[i];

        // Get the attributes thanks to the index
        glm::vec3 vertex = temp_vertices[vertexIndex - 1];
        glm::vec2 uv = temp_uvs[uvIndex - 1];
        glm::vec3 normal = temp_normals[normalIndex - 1];

        // Put the attributes in buffers
        out_vertices.push_back(vertex);
        out_uvs.push_back(uv);
        out_normals.push_back(normal);

    }

    return true;
}

template <class T>
void endswap(T *objp)
{
    unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
    std::reverse(memp, memp + sizeof(T));
}

bool loadCustomRenderFile(const char * path, vector<glm::vec3> & out_atomverts, vector<glm::vec3> & out_atomcols, vector<unsigned short> & out_bondindicies)
{
    printf("Loading render file: %s\n", path);

    ifstream file(path, ios::in | ios::binary | ios::ate);

    if (file.good())
    {
        if (file.is_open())
        {
            //Read all the data into memory
            streampos size = file.tellg();
            file.seekg(0, ios::beg);
            for (int pos = 0; pos < size; pos++)
            {
                char flag;
                file.read(&flag, sizeof(char));
                switch(flag)
                {
                case 'a':
                {
                    float atom[3];
                    float color[3];

                    file.read((char*)&atom, sizeof(float) * 3);
                    glm::vec3 temp(atom[0], atom[1], atom[2]);
                    out_atomverts.push_back(temp);

                    file.read((char*)&color, sizeof(float) * 3);
                    glm::vec3 temp2(color[0], color[1], color[2]);
                    out_atomcols.push_back(temp2);

                    pos += sizeof(float) * 6;
                    break;
                }
                case 'b':
                    int a, b;
                    file.read((char*)&a, sizeof(unsigned short));
                    file.read((char*)&b, sizeof(unsigned short));
                    out_bondindicies.push_back(a);
                    out_bondindicies.push_back(b);
                    pos += sizeof(unsigned short) * 2;
                }
            }
            file.close();
            return true;
        }
        else
        {
            printf("Error: Unable to open input file %s\n", path);
            return false;
        }
    }
    else
    {
        printf("Error: File %s opening is not \"good\".  Is the file open elsewhere?\n", path);
        return false;
    }
}