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

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/GLU.h>

#include "ProteinDisplay.h"
#include "PFDProcessor.h"
#include "displayUtils/shader.h"
#include "displayUtils/controls.h"

using namespace glm;
using namespace std;

GLFWwindow* window;

float dotProd(vec3 & a, vec3 & b)
{
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

vec3 crossProd(vec3 & a, vec3 & b)
{
    return vec3(((a[1] * b[2]) - (a[2] * b[1])), ((a[2] * b[0]) - (a[0] * b[2])), ((a[0] * b[1]) - (a[1] * b[0])));
}

vec3 distance(vec3 & a, vec3 & b)
{
    return vec3((b.x - a.x), (b.y - a.y), (b.z - a.z));
}

float magnitude(vec3 & a)
{
    return sqrtf((a.x * a.x) + (a.y * a.y) + (a.z * a.z));
}


void drawCylinder(vec3 & a, vec3 & b, const float radius, const int divisions)
{
    GLUquadricObj *quadric = gluNewQuadric();
    gluQuadricNormals(quadric, GLU_SMOOTH);
    float vx = a.x - b.x;
    float vy = a.y - b.y;
    float vz = a.z - b.z;

    //handle the degenerate case of z1 == z2 with an approximation
    if (vz == 0)
        vz = .0001;

    float v = sqrt(vx*vx + vy*vy + vz*vz);
    float ax = 57.2957795*acos(vz / v);
    if (vz < 0.0)
        ax = -ax;
    float rx = -vy*vz;
    float ry = vx*vz;
    glPushMatrix();

    //draw the cylinder body
    glTranslatef(a.x, a.y, a.z);
    glRotatef(ax, rx, ry, 0.0);
    gluQuadricOrientation(quadric, GLU_OUTSIDE);
    gluCylinder(quadric, radius, radius, v, divisions, 1);

    //draw the first cap
    gluQuadricOrientation(quadric, GLU_INSIDE);
    gluDisk(quadric, 0.0, radius, divisions, 1);
    glTranslatef(0, 0, v);

    //draw the second cap
    gluQuadricOrientation(quadric, GLU_OUTSIDE);
    gluDisk(quadric, 0.0, radius, divisions, 1);
    glPopMatrix();
    gluDeleteQuadric(quadric);
}


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
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

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

    // Get a handle for our "myTextureSampler" uniform
    GLuint TextureID = glGetUniformLocation(texprogramID, "myTextureSampler");

    //Load everything needed for rendering
    vector<vector<vec3>> helices;
    vector<vector<vec3>> sheets;
    vector<unsigned short> wireindicies;
    vector<vec3> atomverts;
    vector<vec4> atomcols;
    vector<unsigned short> texindicies;
    vector<vec3> texverts;
    vector<vec2> texcoords;
    vector<GLuint> textureIDs;
    PFDReader reader;
    openPFDFileReader(&reader, "test.pfd");
    bool trytexload = loadPFDTextureFile(&reader, atomverts, atomcols, wireindicies, helices, sheets, texverts, textureIDs);
    closePFDFileReader(&reader);

    for (int i = 0; i < textureIDs.size(); i++)
    {
        texindicies.push_back((i * 4));
        texindicies.push_back((i * 4) + 2);
        texindicies.push_back((i * 4) + 1);
        texindicies.push_back((i * 4));
        texindicies.push_back((i * 4) + 3);
        texindicies.push_back((i * 4) + 2);

        
        texcoords.push_back(vec2(1, 0));
        texcoords.push_back(vec2(1, 1));
        texcoords.push_back(vec2(0, 1));
        texcoords.push_back(vec2(0, 0));
        
    }

    setTextureCap(textureIDs.size() - 1);

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
    glBufferData(GL_ARRAY_BUFFER, atomcols.size() * sizeof(vec4), &atomcols[0], GL_STATIC_DRAW);

    GLuint wireelembuff;
    glGenBuffers(1, &wireelembuff);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wireelembuff);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, wireindicies.size() * sizeof(unsigned short), &wireindicies[0], GL_STATIC_DRAW);

    //Setup for dynamic alpha control
    GLint alphaloc = glGetUniformLocation(texprogramID, "inAlpha");
    if (alphaloc == -1)
    {
        printf("Error: Alpha channel not corrently accessed.");
        return 1;
    }

    glUniform1f(alphaloc, 1.0f);
    do {

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);      

        //Draw seconadry structure
        for (int i = 0; i < helices.size(); i++)
        {
            drawCylinder(helices[i][0], helices[i][1], 10.0f, 10);
        }

        // Compute the MVP matrix from keyboard and mouse input
        computeMatricesFromInputs();
        mat4 ProjectionMatrix = getProjectionMatrix();
        mat4 ViewMatrix = getViewMatrix();
        mat4 ModelMatrix = mat4(1.0);
        mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

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
            4,                  // size
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

        // Send our transformation to the currently bound shader, 
        // in the "MVP" uniform
        glUseProgram(texprogramID);

        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
        glUniform1f(alphaloc, getAlphaMod());
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
            6,                  // count
            GL_UNSIGNED_SHORT,   // type
            (void*)(getTextureNum() * sizeof(unsigned short) * 6)           // element array buffer offset
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