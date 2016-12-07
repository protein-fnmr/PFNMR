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
#include "PFDProcessor.h"
#include "displayUtils/shader.h"
#include "displayUtils/texture.h"
#include "displayUtils/controls.h"
#include "displayUtils/objloader.h"
#include "displayUtils/vboindexer.h"

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
    //glEnable(GL_CULL_FACE);

    glDisable(GL_CULL_FACE);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    GLuint texprogramID = LoadShaders("PFNMRtexvert.vertexshader", "PFNMRtexfrag.fragmentshader");
    GLuint wireprogramID = LoadShaders("PFNMRwirevert.vertexshader", "PFNMRwirefrag.fragmentshader");

    // Get a handle for our "MVP" uniform
    GLuint MatrixID = glGetUniformLocation(texprogramID, "MVP");
    GLuint wirematID = glGetUniformLocation(wireprogramID, "MVP");

    // Load the texture
    //GLuint Texture = loadBMP_custom("uvmap.bmp");
    //GLuint Texture = loadDDS("uvmap.dds");

    // Get a handle for our "myTextureSampler" uniform
    GLuint TextureID = glGetUniformLocation(texprogramID, "myTextureSampler");

    // Read our .obj file
    vector<vec3> vertices;
    vector<vec2> uvs;
    vector<vec3> normals; // Won't be used at the moment.
    bool res = loadOBJ("cube.obj", vertices, uvs, normals);

    vector<unsigned short> indices;
    vector<vec3> indexed_vertices;
    vector<vec2> indexed_uvs;
    vector<vec3> indexed_normals;
    indexVBO(vertices, uvs, normals, indices, indexed_vertices, indexed_uvs, indexed_normals);

    // Load it into a VBO
    /*
    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_vertices.size() * sizeof(vec3), &indexed_vertices[0], GL_STATIC_DRAW);

    GLuint uvbuffer;
    glGenBuffers(1, &uvbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_uvs.size() * sizeof(vec2), &indexed_uvs[0], GL_STATIC_DRAW);

    GLuint normalbuffer;
    glGenBuffers(1, &normalbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_normals.size() * sizeof(vec3), &indexed_normals[0], GL_STATIC_DRAW);

    // Generate a buffer for the indices as well
    GLuint elementbuffer;
    glGenBuffers(1, &elementbuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);
    */

    //Load everything needed for rendering
    vector<unsigned short> wireindicies;
    vector<vec3> atomverts;
    vector<vec3> atomcols;
    vector<unsigned short> texindicies;
    vector<vec3> texverts;
    vector<vec2> texcoords;
    vector<GLuint> textureIDs;
    PFDReader reader;
    openPFDFileReader(&reader, "test.pfd");
    bool trytexload = loadPFDTextureFile(&reader, atomverts, atomcols, wireindicies, texverts, textureIDs);
    closePFDFileReader(&reader);

    setTextureCap(textureIDs.size() - 1);

    for (int i = 0; i < textureIDs.size(); i++)
    {
        texindicies.push_back((i * 4));
        texindicies.push_back((i * 4) + 2);
        texindicies.push_back((i * 4) + 1);
        texindicies.push_back((i * 4));
        texindicies.push_back((i * 4) + 3);
        texindicies.push_back((i * 4) + 2);
    }

    texcoords.push_back(vec2(0, 1));
    texcoords.push_back(vec2(1, 1));
    texcoords.push_back(vec2(1, 0));
    texcoords.push_back(vec2(0, 0));

    GLuint texelembuff;
    glGenBuffers(1, &texelembuff);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, texelembuff);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, texindicies.size() * sizeof(unsigned short), &texindicies[0], GL_STATIC_DRAW);

    GLuint texvertsbuffer;
    glGenBuffers(1, &texvertsbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, texvertsbuffer);
    glBufferData(GL_ARRAY_BUFFER, texverts.size() * sizeof(vec3), &texverts[0], GL_STATIC_DRAW);

    GLuint texcoordbuffer;
    glGenBuffers(1, &texcoordbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, texcoordbuffer);
    glBufferData(GL_ARRAY_BUFFER, texcoords.size() * sizeof(vec2), &texcoords[0], GL_STATIC_DRAW);

    GLuint wirevertbuffer;
    glGenBuffers(1, &wirevertbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, wirevertbuffer);
    glBufferData(GL_ARRAY_BUFFER, atomverts.size() * sizeof(vec3), &atomverts[0], GL_STATIC_DRAW);

    GLuint wirecolbuffer;
    glGenBuffers(1, &wirecolbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, wirecolbuffer);
    glBufferData(GL_ARRAY_BUFFER, atomcols.size() * sizeof(vec3), &atomcols[0], GL_STATIC_DRAW);

    GLuint wireelembuff;
    glGenBuffers(1, &wireelembuff);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wireelembuff);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, wireindicies.size() * sizeof(unsigned short), &wireindicies[0], GL_STATIC_DRAW);

    do {

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //Display all the shader stuff
        // Use our shader
        glUseProgram(texprogramID);

        // Compute the MVP matrix from keyboard and mouse input
        computeMatricesFromInputs();
        mat4 ProjectionMatrix = getProjectionMatrix();
        mat4 ViewMatrix = getViewMatrix();
        mat4 ModelMatrix = mat4(1.0);
        mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
        // Send our transformation to the currently bound shader, 
        // in the "MVP" uniform
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureIDs[getTextureNum()]);
        // Set our "myTextureSampler" sampler to user Texture Unit 0
        glUniform1i(TextureID, 0);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, texvertsbuffer);
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
        glBindBuffer(GL_ARRAY_BUFFER, texcoordbuffer);
        glVertexAttribPointer(
            1,                                // attribute
            2,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
        );

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, texelembuff);

        // Draw the triangles !
        glDrawElements(
            GL_TRIANGLES,      // mode
            texindicies.size(),                  // count
            GL_UNSIGNED_SHORT,   // type
            (void*)(0)           // element array buffer offset
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
    glDeleteProgram(texprogramID);
    glDeleteBuffers(1, &texvertsbuffer);
    glDeleteBuffers(1, &texcoordbuffer);
    glDeleteBuffers(1, &texelembuff);
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