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

// C++ code for doing protein displaying
// Made from code taken from http://www.opengl-tutorial.org/, beer is in the mail.

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "ProteinDisplay.h"
#include "displayUtils/shader.h"
#include "displayUtils/texture.h"
#include "displayUtils/controls.h"
#include "displayUtils/objloader.h"
#include "displayUtils/vboindexer.h"

#define THRESHOLDMOD 0.5

using namespace glm;
using namespace std;

GLFWwindow* window;

int ProteinDisplay::initDisplay()
{
    // Initialise GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(1024, 768, "PFNMR Display", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window. Do you have an OpenGL 3.3 capable display?\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited movement
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(window, 1024 / 2, 768 / 2);

    // Dark black background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    GLuint programID = LoadShaders("PFNMRtexvert.vertexshader", "PFNMRtexfrag.fragmentshader");
    GLuint wireprogramID = LoadShaders("PFNMRwirevert.vertexshader", "PFNMRwirefrag.fragmentshader");

    // Get a handle for our "MVP" uniform
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");
    GLuint wirematID = glGetUniformLocation(wireprogramID, "MVP");

    // Load the texture
    //GLuint Texture = loadBMP_custom("uvmap.bmp");
    GLuint Texture = loadDDS("uvmap.dds");

    // Get a handle for our "myTextureSampler" uniform
    GLuint TextureID = glGetUniformLocation(programID, "myTextureSampler");

    // Read our .obj file
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals; // Won't be used at the moment.
    bool res = loadOBJ("cube.obj", vertices, uvs, normals);

    std::vector<unsigned short> indices;
    std::vector<glm::vec3> indexed_vertices;
    std::vector<glm::vec2> indexed_uvs;
    std::vector<glm::vec3> indexed_normals;
    indexVBO(vertices, uvs, normals, indices, indexed_vertices, indexed_uvs, indexed_normals);

    // Load it into a VBO
    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_vertices.size() * sizeof(glm::vec3), &indexed_vertices[0], GL_STATIC_DRAW);

    GLuint uvbuffer;
    glGenBuffers(1, &uvbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_uvs.size() * sizeof(glm::vec2), &indexed_uvs[0], GL_STATIC_DRAW);

    GLuint normalbuffer;
    glGenBuffers(1, &normalbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_normals.size() * sizeof(glm::vec3), &indexed_normals[0], GL_STATIC_DRAW);

    // Generate a buffer for the indices as well
    GLuint elementbuffer;
    glGenBuffers(1, &elementbuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);

    //All the stuff for a test wireframe cube
    std::vector<unsigned short> wireindicies;
    std::vector<glm::vec3> atomverts;
    std::vector<glm::vec3> atomcols;
    res = loadCustomRenderFile("test.pfd", atomverts, atomcols, wireindicies);

    GLuint wirevertbuffer;
    glGenBuffers(1, &wirevertbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, wirevertbuffer);
    glBufferData(GL_ARRAY_BUFFER, atomverts.size() * sizeof(glm::vec3), &atomverts[0], GL_STATIC_DRAW);

    GLuint wirecolbuffer;
    glGenBuffers(1, &wirecolbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, wirecolbuffer);
    glBufferData(GL_ARRAY_BUFFER, atomcols.size() * sizeof(glm::vec3), &atomcols[0], GL_STATIC_DRAW);

    GLuint wireelembuff;
    glGenBuffers(1, &wireelembuff);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wireelembuff);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, wireindicies.size() * sizeof(unsigned short), &wireindicies[0], GL_STATIC_DRAW);

    do {

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //Display all the shader stuff
        // Use our shader
        glUseProgram(programID);

        // Compute the MVP matrix from keyboard and mouse input
        computeMatricesFromInputs();
        glm::mat4 ProjectionMatrix = getProjectionMatrix();
        glm::mat4 ViewMatrix = getViewMatrix();
        glm::mat4 ModelMatrix = glm::mat4(1.0);
        glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

        // Send our transformation to the currently bound shader, 
        // in the "MVP" uniform
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, Texture);
        // Set our "myTextureSampler" sampler to user Texture Unit 0
        glUniform1i(TextureID, 0);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
            0,                  // attribute
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
        );

        // 2nd attribute buffer : UVs
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
        glVertexAttribPointer(
            1,                                // attribute
            2,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
        );

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

        // Draw the triangles !
        glDrawElements(
            GL_TRIANGLES,      // mode
            indices.size(),    // count
            GL_UNSIGNED_SHORT,   // type
            (void*)0           // element array buffer offset
        );

        //Display the wireframe stuff
        glUseProgram(wireprogramID);

        glUniformMatrix4fv(wirematID, 1, GL_FALSE, &MVP[0][0]);

        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, wirevertbuffer);
        glVertexAttribPointer(
            2,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
        );

        glEnableVertexAttribArray(3);
        glBindBuffer(GL_ARRAY_BUFFER, wirecolbuffer);
        glVertexAttribPointer(
            3,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
        );

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wireelembuff);

        glDrawElements(
            GL_LINES,      // mode
            wireindicies.size(),    // count
            GL_UNSIGNED_SHORT,   // type
            (void*)0           // element array buffer offset
        );


        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glDisableVertexAttribArray(3);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

    } // Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
        glfwWindowShouldClose(window) == 0);

    // Cleanup VBO and shader
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteBuffers(1, &uvbuffer);
    glDeleteBuffers(1, &normalbuffer);
    glDeleteBuffers(1, &elementbuffer);
    glDeleteProgram(programID);
    glDeleteTextures(1, &TextureID);
    glDeleteVertexArrays(1, &VertexArrayID);

    glDeleteProgram(wireprogramID);
    glDeleteBuffers(1, &wirecolbuffer);
    glDeleteBuffers(1, &wirevertbuffer);
    glDeleteBuffers(1, &wireelembuff);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}

void ProteinDisplay::makePFD(vector<Atom> & atoms, vector<vector<string>> & colorcsv, const char * outpath)
{
    //Useful numbers for later
    int nAtoms = atoms.size();
    int nColors = colorcsv.size();

    //Setup stuff for file writing
    ofstream file("test.pfd", ios::out | ios::binary);
    if (file.good())
    {

        //Write all the atom coordinates/colors to the file
        printf("Writing atom coordinate data to %s\n", outpath);
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
            file.write(buffer, sizeof(char) + (6 * sizeof(float))); //write the buffer to the file
        }
        //Write all the bond data
        flag = 'b';
        memcpy(buffer, &flag, sizeof(char));
        printf("Writing bond map data to %s\n", outpath);
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
                    file.write(buffer, sizeof(char) + (2 * sizeof(unsigned short)));
                }
            }
        }
        file.close();
    }
    else
    {
        printf("Error: File writing is not \"good\".  Is it open elsewhere?\n");
    }
    printf("Done writing display file %s\n", outpath);
}