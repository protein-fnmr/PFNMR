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

// C++ code for doing controls in the window
// Made from code taken from http://www.opengl-tutorial.org/, beer is in the mail.
#include <GLFW/glfw3.h>
extern GLFWwindow* window; // The "extern" keyword here is to access the variable "window" declared in tutorialXXX.cpp. This is a hack to keep the tutorials simple. Please avoid this.

                           // Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include "controls.h"

glm::mat4 ViewMatrix;
glm::mat4 ProjectionMatrix;

glm::mat4 getViewMatrix() {
    return ViewMatrix;
}
glm::mat4 getProjectionMatrix() {
    return ProjectionMatrix;
}

int textureID = 0;
int cap = 0;

int getTextureNum()
{
    return textureID;
}

void setTextureCap(int val)
{
    cap = val;
}

float alpha = 1.0f;

float getAlphaMod()
{
    return alpha;
}

// Initial position : on +Z
glm::vec3 position = glm::vec3(0, 0, 40);
// Initial horizontal angle : toward -Z
float horizontalAngle = 3.14f;
// Initial vertical angle : none
float verticalAngle = 0.0f;
// Initial Field of View
float initialFoV = 45.0f;

float speed = 10.0f; // 3 units / second
float mouseSpeed = 0.0025f;

bool statelock = true;
bool statelock2 = true;


void computeMatricesFromInputs() {

    // glfwGetTime is called only once, the first time this function is called
    static double lastTime = glfwGetTime();

    // Compute time difference between current and last frame
    double currentTime = glfwGetTime();
    float deltaTime = float(currentTime - lastTime);

    // Get mouse position
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Reset mouse position for next frame
    glfwSetCursorPos(window, 1024 / 2, 768 / 2);

    // Compute new orientation
    horizontalAngle += mouseSpeed * float(1024 / 2 - xpos);
    verticalAngle += mouseSpeed * float(768 / 2 - ypos);

    // Direction : Spherical coordinates to Cartesian coordinates conversion
    glm::vec3 direction(
        cos(verticalAngle) * sin(horizontalAngle),
        sin(verticalAngle),
        cos(verticalAngle) * cos(horizontalAngle)
    );

    // Right vector
    glm::vec3 right = glm::vec3(
        sin(horizontalAngle - 3.14f / 2.0f),
        0,
        cos(horizontalAngle - 3.14f / 2.0f)
    );

    // Up vector
    glm::vec3 up = glm::cross(right, direction);

    // Move forward
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        position += direction * deltaTime * speed;
    }
    // Move backward
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        position -= direction * deltaTime * speed;
    }
    // Strafe right
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        position += right * deltaTime * speed;
    }
    // Strafe left
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        position -= right * deltaTime * speed;
    }

    //Move up
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        position += up * deltaTime * speed;
    }

    //Move down
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        position -= up * deltaTime * speed;
    }

    //Reset the camera
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        position = glm::vec3(0, 0, 40);
        horizontalAngle = 3.14f;
        verticalAngle = 0.0f;
        initialFoV = 45.0f;
    }

    //Change texture number
    if (glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS && statelock) {
        textureID++;
        statelock = !statelock;
    }

    if (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS && statelock) {
        textureID--;
        statelock = !statelock;
    }

    if ((glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_RELEASE) && !statelock) {
        statelock = !statelock;
    }
    if (textureID > cap)
        textureID = cap;
    if (textureID < 0)
        textureID = 0;

    //Change the texture alpha
    if (glfwGetKey(window, GLFW_KEY_KP_9) == GLFW_PRESS && statelock2) {
        alpha += 0.05f;
        statelock2 = !statelock2;
    }

    if (glfwGetKey(window, GLFW_KEY_KP_6) == GLFW_PRESS && statelock2) {
        alpha -= 0.05f;
        statelock2 = !statelock2;
    }

    if ((glfwGetKey(window, GLFW_KEY_KP_9) == GLFW_RELEASE && glfwGetKey(window, GLFW_KEY_KP_6) == GLFW_RELEASE) && !statelock2) {
        statelock2 = !statelock2;
    }
    if (alpha > 1.0f)
        alpha = 1.0f;
    if (alpha < 0.0)
        alpha = 0.0;

    float FoV = initialFoV;// - 5 * glfwGetMouseWheel(); // Now GLFW 3 requires setting up a callback for this. It's a bit too complicated for this beginner's tutorial, so it's disabled instead.

                           // Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    ProjectionMatrix = glm::perspective(FoV, 4.0f / 3.0f, 0.1f, 100.0f);
    // Camera matrix
    ViewMatrix = glm::lookAt(
        position,           // Camera is here
        position + direction, // and looks here : at the same position, plus "direction"
        up                  // Head is up (set to 0,-1,0 to look upside-down)
    );

    // For the next frame, the "last time" will be "now"
    lastTime = currentTime;
}