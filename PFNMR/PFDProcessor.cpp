#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>

#include "Heatmap.h"
#include "GPUTypes.h"
#include "PFDProcessor.h"


#define THRESHOLDMOD 0.5

bool openPFDFileWriter(PFDWriter* writer, string outpath)
{
    writer->file = ofstream(outpath, ios::out | ios::binary);
    if (writer->file.good())
    {
        return true;
    }
    else
    {
        printf("Error: File cannot be opened.  Is it open elsewhere?");
        return false;
    }
}

bool openPFDFileReader(PFDReader* reader, string inpath)
{
    reader->file = ifstream(inpath, ios::in | ios::binary | ios::ate);
    if (reader->file.good())
    {
        return true;
    }
    else
    {
        printf("Error: File cannot be opened.  Is it open elsewhere?");
        return false;
    }
}

void closePFDFileWriter(PFDWriter* writer)
{
    writer->file.close();
}

void closePFDFileReader(PFDReader* reader)
{
    reader->file.close();
}

void writeStructurePFDInfo(PFDWriter* writer, vector<Atom> & atoms, vector<vector<string>> & colorcsv)
{
    //Useful numbers for later
    int nAtoms = atoms.size();
    int nColors = colorcsv.size();
    //Write all the atom coordinates/colors to the file
    char buffer[25]; //setup the buffer
    char flag = 'a'; //set the data type flag
    memcpy(buffer, &flag, sizeof(char));
    for (int i = 0; i < nAtoms; i++)
    {
        string elem = atoms[i].element;

        float cr = 0, cg = 0, cb = 0;
        for (int j = 0; j < nColors; j++)
        {
            if (colorcsv[j][0] == elem)
            {
                cr = stof(colorcsv[j][1]);
                cg = stof(colorcsv[j][2]);
                cb = stof(colorcsv[j][3]);
                break;
            }
        }
        memcpy(&buffer[sizeof(char)], &atoms[i].x, sizeof(float));
        memcpy(&buffer[sizeof(char) + sizeof(float)], &atoms[i].y, sizeof(float));
        memcpy(&buffer[sizeof(char) + (2 * sizeof(float))], &atoms[i].z, sizeof(float));
        memcpy(&buffer[sizeof(char) + (3 * sizeof(float))], &cr, sizeof(float));
        memcpy(&buffer[sizeof(char) + (4 * sizeof(float))], &cg, sizeof(float));
        memcpy(&buffer[sizeof(char) + (5 * sizeof(float))], &cb, sizeof(float));
        writer->file.write(buffer, sizeof(char) + (6 * sizeof(float))); //write the buffer to the file
    }
    //Write all the bond data
    flag = 'b';
    memcpy(buffer, &flag, sizeof(char));
    for (unsigned short i = 0; i < nAtoms; i++)
    {
        for (unsigned short j = 0; j < i; j++)
        {
            float diffx = atoms[i].x - atoms[j].x;
            float diffy = atoms[i].y - atoms[j].y;
            float diffz = atoms[i].z - atoms[j].z;
            float distance = sqrtf((diffx * diffx) + (diffy * diffy) + (diffz * diffz));
            float threshold = (atoms[i].vdw + atoms[i].vdw) * THRESHOLDMOD;

            if (distance <= threshold)
            {
                memcpy(&buffer[sizeof(char)], &i, sizeof(unsigned short));
                memcpy(&buffer[sizeof(char) + sizeof(unsigned short)], &j, sizeof(unsigned short));
                writer->file.write(buffer, sizeof(char) + (2 * sizeof(unsigned short)));
            }
        }
    }
}

void writeSecondaryStructureData(PFDWriter* writer, vector<vector<Atom>> & helices, vector<vector<Atom>> & sheets)
{
    char buffer[25];
    char flag = 'h';
    memcpy(&buffer, &flag, sizeof(char));
    for (int i = 0; i < helices.size(); i++)
    {
        memcpy(&buffer[sizeof(char)], &helices[i][0].x, sizeof(float));
        memcpy(&buffer[sizeof(char) + sizeof(float)], &helices[i][0].y, sizeof(float));
        memcpy(&buffer[sizeof(char) + (2 * sizeof(float))], &helices[i][0].z, sizeof(float));
        memcpy(&buffer[sizeof(char) + (3 * sizeof(float))], &helices[i][1].x, sizeof(float));
        memcpy(&buffer[sizeof(char) + (4 * sizeof(float))], &helices[i][1].y, sizeof(float));
        memcpy(&buffer[sizeof(char) + (5 * sizeof(float))], &helices[i][1].z, sizeof(float));
        writer->file.write(buffer, sizeof(char) + (6 * sizeof(float))); //write the buffer to the file
    }
    flag = 's';
    memcpy(&buffer, &flag, sizeof(char));
    for (int i = 0; i < sheets.size(); i++)
    {
        memcpy(&buffer[sizeof(char)], &sheets[i][0].x, sizeof(float));
        memcpy(&buffer[sizeof(char) + sizeof(float)], &sheets[i][0].y, sizeof(float));
        memcpy(&buffer[sizeof(char) + (2 * sizeof(float))], &sheets[i][0].z, sizeof(float));
        memcpy(&buffer[sizeof(char) + (3 * sizeof(float))], &sheets[i][1].x, sizeof(float));
        memcpy(&buffer[sizeof(char) + (4 * sizeof(float))], &sheets[i][1].y, sizeof(float));
        memcpy(&buffer[sizeof(char) + (5 * sizeof(float))], &sheets[i][1].z, sizeof(float));
        writer->file.write(buffer, sizeof(char) + (6 * sizeof(float))); //write the buffer to the file
    }

}

//[flag][side resolution][plane dimensions]{[data]}
void writeDielectricFrameData(PFDWriter* writer, const uint8_t* image, vector<float> & planeDims, uint32_t imgSideResolution)
{
    auto sidesqdata = imgSideResolution * imgSideResolution;
    char buffer[29];
    char flag = 'd'; //set the data type flag
    memcpy(&buffer, &flag, sizeof(char));
    memcpy(&buffer[sizeof(char)], &imgSideResolution, sizeof(uint32_t));
    memcpy(&buffer[sizeof(char) + sizeof(uint32_t)], &planeDims[0], sizeof(float) * 6);
    writer->file.write(buffer, sizeof(char) + sizeof(uint32_t) + (sizeof(float) * 6));
    for (int i = 0; i < sidesqdata; i++)
    {
        memcpy(&buffer, &image[sizeof(uint8_t) * 4 * i], sizeof(uint8_t) * 4);
        writer->file.write(buffer, sizeof(uint8_t) * 4);
    }
}

//[flag][num frames][side resolution][heatmap range]{[framedata]}
//returns streampos of the header for rewriting
streampos writeHeatmapSetHeader(PFDWriter *writer, short numframes, uint32_t imgSideResolution, vector<float> heatmaprange)
{
    auto pos = writer->file.tellp();
    char buffer[29];
    char flag = 'm';
    memcpy(&buffer, &flag, sizeof(char));
    memcpy(&buffer[sizeof(char)], &numframes, sizeof(short));
    memcpy(&buffer[sizeof(char) + sizeof(short)], &imgSideResolution, sizeof(uint32_t));
    memcpy(&buffer[sizeof(char) + sizeof(short) + sizeof(uint32_t)], &heatmaprange[0], sizeof(float) * 2);
    writer->file.write(buffer, sizeof(char) + sizeof(short) + sizeof(uint32_t) + (sizeof(float) * 2));
    return pos; 
}

//[flag][num frames][side resolution][heatmap range]{[framedata]}
//writes to the specified position, then hops back to the original position
void rewriteHeatmapSetHeader(PFDWriter *writer, short numframes, uint32_t imgSideResolution, vector<float> heatmaprange, streampos pos)
{
    auto origpos = writer->file.tellp();
    writer->file.seekp(pos);
    char buffer[29];
    char flag = 'm';
    memcpy(&buffer, &flag, sizeof(char));
    memcpy(&buffer[sizeof(char)], &numframes, sizeof(short));
    memcpy(&buffer[sizeof(char) + sizeof(short)], &imgSideResolution, sizeof(uint32_t));
    memcpy(&buffer[sizeof(char) + sizeof(short) + sizeof(uint32_t)], &heatmaprange[0], sizeof(float) * 2);
    writer->file.write(buffer, sizeof(char) + sizeof(short) + sizeof(uint32_t) + (sizeof(float) * 2));
    writer->file.seekp(origpos);
}

//[planedims][data]
void writeHeatmapFrameData(PFDWriter* writer, const float* image, vector<float> & planeDims, uint32_t imgSideResolution)
{
    auto sidesqdata = imgSideResolution * imgSideResolution;
    char buffer[29];
    memcpy(&buffer, &planeDims[0], sizeof(float) * 6);
    writer->file.write(buffer, (sizeof(float) * 6));
    for (int i = 0; i < sidesqdata; i++)
    {
        memcpy(&buffer, &image[i], sizeof(float));
        writer->file.write(buffer, sizeof(float));
    }
}

bool loadPFDFile(PFDReader* reader, vector<glm::vec3> & out_atomverts, vector<glm::vec3> & out_atomcols, vector<unsigned short> & out_bondindicies, 
    vector<vector<glm::vec3>> & out_helices, vector<vector<glm::vec3>> & out_sheets)
{
    if (reader->file.is_open())
    {
        //Read all the data into memory
        streampos size = reader->file.tellg();
        reader->file.seekg(0, ios::beg);
        for (int pos = 0; pos < size; pos++)
        {
            char flag;
            reader->file.read(&flag, sizeof(char));
            switch (flag)
            {
            case 'a':
            {
                float atom[3];
                float color[3];

                reader->file.read((char*)&atom, sizeof(float) * 3);
                glm::vec3 temp(atom[0], atom[1], atom[2]);
                out_atomverts.push_back(temp);

                reader->file.read((char*)&color, sizeof(float) * 3);
                glm::vec3 temp2(color[0], color[1], color[2]);
                out_atomcols.push_back(temp2);

                pos += sizeof(float) * 6;
                break;
            }
            case 'b':
                int a, b;
                reader->file.read((char*)&a, sizeof(unsigned short));
                reader->file.read((char*)&b, sizeof(unsigned short));
                out_bondindicies.push_back(a);
                out_bondindicies.push_back(b);
                pos += sizeof(unsigned short) * 2;
                break;
            case 'h':
            {
                float atoms[6];
                reader->file.read((char*)&atoms, sizeof(float) * 6);
                glm::vec3 temp(atoms[0], atoms[1], atoms[2]);
                glm::vec3 temp2(atoms[3], atoms[4], atoms[5]);
                out_helices.push_back({ temp , temp2 });
                break;
            }
            case 's':
            {
                float atoms[6];
                reader->file.read((char*)&atoms, sizeof(float) * 6);
                glm::vec3 temp(atoms[0], atoms[1], atoms[2]);
                glm::vec3 temp2(atoms[3], atoms[4], atoms[5]);
                out_sheets.push_back({ temp , temp2 });
                break;
            }
            default:
                printf("ERROR: Encountered an unknown flag %c.  Ending reading procedure...", flag);
                return false;
            }
        }
        return true;
    }
    else
    {
        printf("ERROR: File is already open!");
        return false;
    }
}

bool loadPFDTextureFile(PFDReader* reader, vector<glm::vec3> & out_atomverts, vector<glm::vec4> & out_atomcols, vector<unsigned short> & out_bondindicies,
    vector<vector<glm::vec3>> & out_helices, vector<vector<glm::vec3>> & out_sheets, vector<glm::vec3> & out_texverts, vector<GLuint> & texIDs)
{
    if (reader->file.is_open())
    {
        //Read all the data into memory
        streampos size = reader->file.tellg();
        reader->file.seekg(0, ios::beg);
        int slices = 0;
        for (int pos = 0; pos < size; pos++)
        {
            char flag;
            reader->file.read(&flag, sizeof(char));
            switch (flag)
            {
            case 'a':
            {
                float atom[3];
                float color[3];

                reader->file.read((char*)&atom, sizeof(float) * 3);
                glm::vec3 temp(atom[0], atom[1], atom[2]);
                out_atomverts.push_back(temp);

                reader->file.read((char*)&color, sizeof(float) * 3);
                glm::vec4 temp2(color[0], color[1], color[2], 1.0f);
                out_atomcols.push_back(temp2);

                pos += sizeof(float) * 6;
                break;
            }
            case 'b':
                int a, b;
                reader->file.read((char*)&a, sizeof(unsigned short));
                reader->file.read((char*)&b, sizeof(unsigned short));
                out_bondindicies.push_back(a);
                out_bondindicies.push_back(b);
                pos += sizeof(unsigned short) * 2;
                break;
            case 'h':
            {
                float atoms[6];
                reader->file.read((char*)&atoms, sizeof(float) * 6);
                glm::vec3 temp(atoms[0], atoms[1], atoms[2]);
                glm::vec3 temp2(atoms[3], atoms[4], atoms[5]);
                out_helices.push_back({ temp , temp2 });
                pos += sizeof(float) * 6;
                break;
            }
            case 's':
            {
                float atoms[6];
                reader->file.read((char*)&atoms, sizeof(float) * 6);
                glm::vec3 temp(atoms[0], atoms[1], atoms[2]);
                glm::vec3 temp2(atoms[3], atoms[4], atoms[5]);
                out_sheets.push_back({ temp , temp2 });
                pos += sizeof(float) * 6;
                break;
            }
            case 'd':
            {
                uint32_t sideres;
                reader->file.read((char*)&sideres, sizeof(uint32_t));
                auto sidesq = sideres * sideres;

                //Read and generate the appropriate texture verticies
                float temp[6];
                reader->file.read((char*)&temp, sizeof(float) * 6);
                out_texverts.push_back(glm::vec3(temp[0], temp[1], temp[5]));
                out_texverts.push_back(glm::vec3(temp[3], temp[4], temp[5]));
                out_texverts.push_back(glm::vec3(temp[3], temp[4], temp[2]));
                out_texverts.push_back(glm::vec3(temp[0], temp[1], temp[2]));

                //Read image information into texture ID
                GLuint textureID;
                glGenTextures(1, &textureID);
                glBindTexture(GL_TEXTURE_2D, textureID);
                unsigned char * data;
                data = new unsigned char[sidesq * 4 * sizeof(char)];
                reader->file.read((char*)data, sidesq * 4 * sizeof(char));
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sideres, sideres, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
                delete[] data;

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                glGenerateMipmap(GL_TEXTURE_2D);

                texIDs.push_back(textureID);

                pos += sizeof(uint32_t) + (sizeof(float) * 6) + (sizeof(char) * sidesq * 4);
                break;
            }
            case 'm':
            {
                short numframes;
                uint32_t sideres;
                float hmmin, hmmax;

                //Get out the important header information
                reader->file.read((char*)&numframes, sizeof(short));
                reader->file.read((char*)&sideres, sizeof(uint32_t));
                auto sidesq = sideres * sideres;
                reader->file.read((char*)&hmmin, sizeof(float));
                reader->file.read((char*)&hmmax, sizeof(float));
                float hmrange = (hmmax - hmmin);

                //Read each texture frame
                for (int i = 0; i < numframes; i++)
                {
                    //Get the vertex position information
                    float temp[6];
                    reader->file.read((char*)&temp, sizeof(float) * 6);
                    out_texverts.push_back(glm::vec3(temp[0], temp[1], temp[5]));
                    out_texverts.push_back(glm::vec3(temp[3], temp[4], temp[5]));
                    out_texverts.push_back(glm::vec3(temp[3], temp[4], temp[2]));
                    out_texverts.push_back(glm::vec3(temp[0], temp[1], temp[2]));

                    GLuint textureID;
                    glGenTextures(1, &textureID);
                    glBindTexture(GL_TEXTURE_2D, textureID);

                    //Start reading off the data
                    float * data;
                    data = new float[sidesq * 4 * sizeof(float)];
                    for (int j = 0; j < sidesq; j++)
                    {
                        float tempval;
                        reader->file.read((char*)&tempval, sizeof(float)); //Read out a data value
                        auto hmval = 1.0f - ((tempval - hmmin) / hmrange); //Calculate the heatmap value
                        auto hmrgb = getHeatMapColor(hmval);
                        memcpy(&data[j * 4], hmrgb, sizeof(float) * 3);
                        data[j * 4 + 3] = 1.0f;
                    }
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sideres, sideres, 0, GL_RGBA, GL_FLOAT, data);
                    delete[] data;

                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                    glGenerateMipmap(GL_TEXTURE_2D);

                    texIDs.push_back(textureID);                    
                }
                pos += sizeof(short) + sizeof(uint32_t) + (sizeof(float) * 2) + (numframes * ((sizeof(float) * (6 + sidesq))));
                break;
            }
            default:
                printf("ERROR: Encountered an unknown flag %c.  Ending reading procedure...", flag);
                return false;
            }
        }
        return true;
    }
    else
    {
        printf("ERROR: File is already open!");
        return false;
    }
}