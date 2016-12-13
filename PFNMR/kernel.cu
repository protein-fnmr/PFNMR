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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// use the math constants
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <locale>
#include <tuple>
#include <math.h>

#include "GPUTypes.h"

using namespace std;

// the density kernel that is ran on the GPU
__global__ void sliceDensityKernel(float *out, const GPUAtom *inAtoms, const GridPoint *inGrid,
    const float variance, const size_t nAtoms, const size_t nGridPoints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nGridPoints && j < nAtoms)
    {
        float diffx = inGrid[i].x - inAtoms[j].x;
        float diffy = inGrid[i].y - inAtoms[j].y;
        float diffz = inGrid[i].z - inAtoms[j].z;
        float distance = (diffx * diffx) + (diffy * diffy) + (diffz * diffz);

        out[(j * nGridPoints) + i] = 1.0f - expf((-1.0f * distance) / ((variance * variance) * (inAtoms[j].vdw * inAtoms[j].vdw)));
    }
}

// the dielectric kernel that is ran on the GPU
__global__ void sliceDielectricKernel(float *out, const float *inDensity, const float refDielectric,
    const float outdielectric, const size_t nAtoms, const size_t nGridPoints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float moldensity = 1.0f;

    if (i < nGridPoints)
    {
        for (int j = 0; j < nAtoms; ++j)
            moldensity *= inDensity[(j * nGridPoints) + i];

        out[i] = ((1.0f - moldensity) * refDielectric) + (moldensity * outdielectric);
    }
}

// set up the gpu for the density calculations
cudaError_t sliceDensityCuda(float *out, const GPUAtom *inAtoms, const GridPoint *inGrid,
    const float variance, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp)
{
    // define device arrays
    GPUAtom *dev_atom = 0;
    GridPoint *dev_grid = 0;
    float *dev_out = 0;
    cudaError_t cudaStatus;

	// find the most effective dimensions for our calculations
	int blockDim = sqrt(deviceProp.maxThreadsPerBlock);
	auto blockSize = dim3(blockDim, blockDim);
	auto gridSize = dim3(round((blockDim - 1 + nGridPoints) / blockDim), round((blockDim - 1 + nAtoms) / blockDim));

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nAtoms * nGridPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_atom, nAtoms * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_grid, nGridPoints * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_atom, inAtoms, nAtoms * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_grid, inGrid, nGridPoints * sizeof(GridPoint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // Launch a kernel on the GPU.
    sliceDensityKernel <<<gridSize, blockSize>>> (dev_out, dev_atom, dev_grid, variance, nAtoms, nGridPoints);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "density kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching density kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nAtoms * nGridPoints * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // clear all our device arrays
Error:
    cudaFree(dev_atom);
    cudaFree(dev_grid);
    cudaFree(dev_out);

    return cudaStatus;
}

// set up the gpu for the dielectric calculations
cudaError_t sliceDielectricCuda(float *out, const float *in, const float refDielectric,
    const float outdielectric, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp)
{
    // the device arrays
    float *dev_in = 0;
    float *dev_out = 0;
    cudaError_t cudaStatus;

	// use div because it's more accurrate than the rounding BS
	auto gridDiv = div(nGridPoints, deviceProp.maxThreadsPerBlock);
	auto gridY = gridDiv.quot;

	// ass backwards way of rounding up (maybe use the same trick as above? It might be "faster")
	if (gridDiv.rem != 0)
		gridY++;

	// find the block and grid size
	auto blockSize = deviceProp.maxThreadsPerBlock;
	int gridSize = min(16 * deviceProp.multiProcessorCount, gridY);

    // Allocate GPU buffers for vectors
    cudaStatus = cudaMalloc((void**)&dev_out, nGridPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_in, nAtoms * nGridPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, in, nAtoms * nGridPoints * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // Launch a kernel on the GPU.
    sliceDielectricKernel <<<gridSize, blockSize>>> (dev_out, dev_in, refDielectric, outdielectric, nAtoms, nGridPoints);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "dielectric kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching density kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nGridPoints * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // delete all our device arrays
Error:
    cudaFree(dev_in);
    cudaFree(dev_out);

    return cudaStatus;
}

//===========================================KERNEL STUFF FOR RAPID ELECTRIC FIELD CALCULATIONS===========================================
__global__ void eFieldDensityKernel(float *out, float *xspans, const GPUChargeAtom *inAtoms, const GPUEFP efp,
    const float variance, const size_t offset, const size_t resopsperiter, const size_t nAtoms, const size_t resolution)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //Current position in the multi-atom resolution strip currently being processed
    int j = blockIdx.y * blockDim.y + threadIdx.y; //Current atom for density calculation
    int resopspos = (i + offset); //Where we are along the global multi-atom resolution strip
    if ((i < resopsperiter) && (j < nAtoms) && (resopspos < (resolution * nAtoms)))
    {
        int fieldAtomIndex = resopspos / resolution;  //Get which atoms EField we are working on
        float posWithinFAStrip = resopspos % resolution;  //Get what the relative position index is from the FA to the EFP

        float percent = (posWithinFAStrip + 1.0f) / (resolution + 1.0f); //Relative position to be calculated (This will be provided later for Gauss Quadrature methodology)

        //Calculate relevant distance calculation.  
        float diffx = (percent * (efp.x - inAtoms[fieldAtomIndex].x)) + inAtoms[fieldAtomIndex].x - inAtoms[j].x;
        float diffy = (percent * (efp.y - inAtoms[fieldAtomIndex].y)) + inAtoms[fieldAtomIndex].y - inAtoms[j].y;
        float diffz = (percent * (efp.z - inAtoms[fieldAtomIndex].z)) + inAtoms[fieldAtomIndex].z - inAtoms[j].z;
        float distance = (diffx * diffx) + (diffy * diffy) + (diffz * diffz);

        //Calculate the density and and store it.   
        out[(resopsperiter * j) + i] = 1.0f - expf((-1.0f * distance) / ((variance * variance) * (inAtoms[j].vdw * inAtoms[j].vdw)));

        if (posWithinFAStrip == 0) //If we are at the closest point to the atom, report the distance for the future integration calculation
        {
            float linediffx = efp.x - inAtoms[fieldAtomIndex].x;
            float linediffy = efp.y - inAtoms[fieldAtomIndex].y;
            float linediffz = efp.z - inAtoms[fieldAtomIndex].z;
            float linedistance = sqrtf((linediffx * linediffx) + (linediffy * linediffy) + (linediffz * linediffz));
            xspans[fieldAtomIndex] = linedistance / (resolution + 1.0f);
        }
    }
}

__global__ void eFieldDielectricKernel(float *out, const float *inDensity, const float innerdielectric,
    const float outerdielectric, const size_t offset, const size_t resopsperiter, const size_t nAtoms,
    const size_t resolution)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int resopspos = (i + offset);
    float moldensity = 1.0f;
    if ((i < resopsperiter) && (resopspos < (resolution * nAtoms)))
    {
        int curratom = resopspos / resolution;
        int currres = resopspos % resolution;
        for (int j = 0; j < nAtoms; j++)
            moldensity *= inDensity[(j * resopsperiter) + i];

        out[(currres * nAtoms) + curratom] = ((1.0f - moldensity) * innerdielectric) + (moldensity * outerdielectric);
    }
}

__global__ void trapIntegrationKernel(float *out, const float *inXSpans, const float *inY, const size_t nStrips, const size_t nPoints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nStrips)
    {
        float value = 0.0f;
        for (int j = 1; j < nPoints; j++)
        {
            value += inXSpans[i] * ((inY[(j * nStrips) + i] + inY[((j - 1) * nStrips) + i]) / 2.0f);
        }
        out[i] = value;
    }
}

__global__ void sqrtf2DKernel(float *out, const size_t nX, const size_t nY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nX && j < nY)
    {
        out[(j * nX) + i] = sqrtf(out[(j * nX) + i]);
    }
}

__global__ void electricFieldComponentKernel(GPUEFP *out, const float *inEffLengths, const GPUChargeAtom *inAtoms, const float coulconst, const size_t nEFPs, const size_t nAtoms)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < nEFPs)
    {
        float fieldx = 0.0f;
        float fieldy = 0.0f;
        float fieldz = 0.0f;
        for (int i = 0; i < nAtoms; i++)
        {
            if (inAtoms[i].resid != out[j].resid || inAtoms[i].chainid != out[j].chainid) //Ignore all atoms in residue (These are handles quantum mechanically)
            {
                //Get distance parameters
                float diffx = inAtoms[i].x - out[j].x;
                if (diffx == 0.0)
                    diffx = FLT_EPSILON;
                float diffy = inAtoms[i].y - out[j].y;
                float diffz = inAtoms[i].z - out[j].z;
                float distance = sqrtf((diffx * diffx) + (diffy * diffy) + (diffz * diffz));

                //Get orientation parameters
                float Etot = (inAtoms[i].charge * coulconst) / (inEffLengths[(j*nAtoms) + i] * inEffLengths[(j*nAtoms) + i]);
                float theta = asinf(diffy / distance);
                float phi = atanf(diffz / diffx);
                if (diffx < 0.0f)
                    phi += M_PI;

                //Calculate and add the field components
                fieldx += Etot * cosf(theta) * cosf(phi);
                fieldy += Etot * sinf(theta);
                fieldz += Etot * cosf(theta) * sinf(phi);
            }
        }
        out[j].fieldx = fieldx;
        out[j].fieldy = fieldy;
        out[j].fieldz = fieldz;
    }
}

cudaError_t eFieldDensityCuda(float *out, float *xspans, const GPUChargeAtom *inAtoms, const GPUEFP efp,
    const float variance, const size_t offset, const size_t resopsperiter, const size_t nAtoms,
    const size_t resolution, cudaDeviceProp &deviceProp)
{
    // define device arrays
    GPUChargeAtom *dev_atom = 0;
    GPUEFP *dev_EFP = 0;
    float *dev_out = 0;
    float *dev_xspans = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, resopsperiter * nAtoms * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_xspans, nAtoms * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_atom, nAtoms * sizeof(GPUChargeAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_atom, inAtoms, nAtoms * sizeof(GPUChargeAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_xspans, xspans, nAtoms * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // find the most effective dimensions for our calculations
    int blockDim = sqrt(deviceProp.maxThreadsPerBlock);
    auto blockSize = dim3(blockDim, blockDim);
    auto gridSize = dim3(round((blockDim - 1 + resopsperiter) / blockDim), round((blockDim - 1 + nAtoms) / blockDim));

    // Launch a kernel on the GPU.
    eFieldDensityKernel << <gridSize, blockSize >> > (dev_out, dev_xspans, dev_atom, efp, variance, offset, resopsperiter, nAtoms, resolution);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "density kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching density kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nAtoms * resopsperiter * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(xspans, dev_xspans, nAtoms * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // clear all our device arrays
Error:
    cudaFree(dev_atom);
    cudaFree(dev_EFP);
    cudaFree(dev_out);

    return cudaStatus;
}

cudaError_t eFieldDielectricCuda(float *out, const float *inDensity, const float innerdielectric,
    const float outerdielectric, const size_t offset, const size_t resopsperiter, const size_t nAtoms,
    const size_t resolution, cudaDeviceProp &deviceProp)
{
    // the device arrays
    float *dev_in = 0;
    float *dev_out = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors
    cudaStatus = cudaMalloc((void**)&dev_out, resolution * nAtoms * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_in, nAtoms * resopsperiter * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, inDensity, nAtoms * resopsperiter * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_out, out, nAtoms * resolution * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // use div because it's more accurrate than the rounding BS
    auto gridDiv = div(resopsperiter, deviceProp.maxThreadsPerBlock);
    auto gridY = gridDiv.quot;

    // ass backwards way of rounding up (maybe use the same trick as above? It might be "faster")
    if (gridDiv.rem != 0)
        gridY++;

    // find the block and grid size
    auto blockSize = deviceProp.maxThreadsPerBlock;
    int gridSize = min(16 * deviceProp.multiProcessorCount, gridY);

    // Launch a kernel on the GPU.
    eFieldDielectricKernel << <gridSize, blockSize >> > (dev_out, dev_in, innerdielectric, outerdielectric, offset, resopsperiter, nAtoms, resolution);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "dielectric kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching dielectric kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nAtoms * resolution * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // delete all our device arrays
Error:
    cudaFree(dev_in);
    cudaFree(dev_out);

    return cudaStatus;
}

cudaError_t trapIntegrationCuda(float *out, const float *inXSpans, const float *inY, const size_t nStrips,
    const size_t nPoints, cudaDeviceProp &deviceProp)
{
    // define device arrays
    float *dev_inXSpans = 0;
    float *dev_inY = 0;
    float *dev_out = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nStrips * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inXSpans, nStrips * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inY, nStrips * nPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_inXSpans, inXSpans, nStrips * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inY, inY, nStrips * nPoints * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // use div because it's more accurrate than the rounding BS
    auto gridDiv = div(nStrips, deviceProp.maxThreadsPerBlock);
    auto gridY = gridDiv.quot;

    // ass backwards way of rounding up (maybe use the same trick as above? It might be "faster")
    if (gridDiv.rem != 0)
        gridY++;

    // find the block and grid size
    auto blockSize = deviceProp.maxThreadsPerBlock;
    int gridSize = min(16 * deviceProp.multiProcessorCount, gridY);

    // Launch a kernel on the GPU.
    trapIntegrationKernel << <gridSize, blockSize >> > (dev_out, dev_inXSpans, dev_inY, nStrips, nPoints);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "trapezoid integration kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching trapezoid integration kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nStrips * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // delete all our device arrays
Error:
    cudaFree(dev_inXSpans);
    cudaFree(dev_inY);
    cudaFree(dev_out);

    return cudaStatus;
}

cudaError_t sqrtf2DCuda(float *out, const size_t nX, const size_t nY, cudaDeviceProp &deviceProp)
{
    // define device arrays
    float *dev_out = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nX * nY * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_out, out, nX * nY * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // find the most effective dimensions for our calculations
    int blockDim = sqrt(deviceProp.maxThreadsPerBlock);
    auto blockSize = dim3(blockDim, blockDim);
    auto gridSize = dim3(round((blockDim - 1 + nX) / blockDim), round((blockDim - 1 + nY) / blockDim));

    // Launch a kernel on the GPU.
    sqrtf2DKernel << <gridSize, blockSize >> > (dev_out, nX, nY);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "sqrtf2D kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching sqrtf2D kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nX * nY * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // clear all our device arrays
Error:
    cudaFree(dev_out);

    return cudaStatus;
}

cudaError_t electricFieldComponentCuda(GPUEFP *out, const float *inEffLengths, const GPUChargeAtom *inAtoms,
    const float coulconst, const size_t nEFPs, const size_t nAtoms, cudaDeviceProp &deviceProp)
{
    // define device arrays
    GPUEFP *dev_out = 0;
    float *dev_inEffLengths = 0;
    GPUChargeAtom *dev_inAtoms = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nEFPs * sizeof(GPUEFP));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inEffLengths, nAtoms * nEFPs * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inAtoms, nAtoms * sizeof(GPUChargeAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_out, out, nEFPs * sizeof(GPUEFP), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inEffLengths, inEffLengths, nAtoms * nEFPs * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inAtoms, inAtoms, nAtoms * sizeof(GPUChargeAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // find the most effective dimensions for our calculations
    // use div because it's more accurrate than the rounding BS
    auto gridDiv = div(nEFPs, deviceProp.maxThreadsPerBlock);
    auto gridY = gridDiv.quot;

    // ass backwards way of rounding up (maybe use the same trick as above? It might be "faster")
    if (gridDiv.rem != 0)
        gridY++;

    // find the block and grid size
    auto blockSize = deviceProp.maxThreadsPerBlock;
    int gridSize = min(16 * deviceProp.multiProcessorCount, gridY);

    // Launch a kernel on the GPU.
    electricFieldComponentKernel << <gridSize, blockSize >> > (dev_out, dev_inEffLengths, dev_inAtoms, coulconst, nEFPs, nAtoms);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "electric field component kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching electric field component kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nEFPs * sizeof(GPUEFP), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // clear all our device arrays
Error:
    cudaFree(dev_out);

    return cudaStatus;
}

//===========================================KERNEL STUFF FOR GAUSS QUAD INTEGRATION===========================================
__global__ void eFieldDensityGQKernel(float *out, float *xspans, const GPUChargeAtom *inAtoms, const float *inAbsci, const GPUEFP efp,
    const float variance, const size_t offset, const size_t resopsperiter, const size_t nAtoms, const size_t resolution)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //Current position in the multi-atom resolution strip currently being processed
    int j = blockIdx.y * blockDim.y + threadIdx.y; //Current atom for density calculation
    int resopspos = (i + offset); //Where we are along the global multi-atom resolution strip
    if ((i < resopsperiter) && (j < nAtoms) && (resopspos < (resolution * nAtoms)))
    {
        int fieldAtomIndex = resopspos / resolution;  //Get which atoms EField we are working on
        int posWithinFAStrip = resopspos % resolution;  //Get what the relative position index is from the FA to the EFP

                                                        //Calculate relevant distance calculation.  
        float diffx = ((((efp.x - inAtoms[fieldAtomIndex].x) / 2.0f) * inAbsci[posWithinFAStrip]) + ((efp.x + inAtoms[fieldAtomIndex].x) / 2.0f)) - inAtoms[j].x;
        float diffy = ((((efp.y - inAtoms[fieldAtomIndex].y) / 2.0f) * inAbsci[posWithinFAStrip]) + ((efp.y + inAtoms[fieldAtomIndex].y) / 2.0f)) - inAtoms[j].y;
        float diffz = ((((efp.z - inAtoms[fieldAtomIndex].z) / 2.0f) * inAbsci[posWithinFAStrip]) + ((efp.z + inAtoms[fieldAtomIndex].z) / 2.0f)) - inAtoms[j].z;
        float distance = (diffx * diffx) + (diffy * diffy) + (diffz * diffz);

        //Calculate the density and and store it.   
        out[(resopsperiter * j) + i] = 1.0f - expf((-1.0f * distance) / ((variance * variance) * (inAtoms[j].vdw * inAtoms[j].vdw)));

        if (posWithinFAStrip == 0) //If we are at the closest point to the atom, report the distance for the future integration calculation
        {
            float linediffx = efp.x - inAtoms[fieldAtomIndex].x;
            float linediffy = efp.y - inAtoms[fieldAtomIndex].y;
            float linediffz = efp.z - inAtoms[fieldAtomIndex].z;
            float linedistance = sqrtf((linediffx * linediffx) + (linediffy * linediffy) + (linediffz * linediffz));
            xspans[fieldAtomIndex] = linedistance / 2.0f;
        }
    }
}

__global__ void gaussQuadIntegrationKernel(float *out, const float *inXSpans, const float *inY, const float *inWeights, const size_t nStrips, const size_t nPoints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nStrips)
    {
        float value = 0.0f;
        for (int j = 0; j < nPoints; j++)
        {
            value += inWeights[j] * inY[(j * nStrips) + i];
        }
        out[i] = inXSpans[i] * value;
    }
}

cudaError_t eFieldDensityGQCuda(float *out, float *xspans, const GPUChargeAtom *inAtoms, const float *inAbsci, const GPUEFP efp,
    const float variance, const size_t offset, const size_t resopsperiter, const size_t nAtoms,
    const size_t resolution, cudaDeviceProp &deviceProp)
{
    // define device arrays
    GPUChargeAtom *dev_atom = 0;
    GPUEFP *dev_EFP = 0;
    float *dev_out = 0;
    float *dev_xspans = 0;
    float *dev_absci = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, resopsperiter * nAtoms * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_xspans, nAtoms * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_atom, nAtoms * sizeof(GPUChargeAtom));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_absci, resolution * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_atom, inAtoms, nAtoms * sizeof(GPUChargeAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_xspans, xspans, nAtoms * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }


    cudaStatus = cudaMemcpy(dev_absci, inAbsci, resolution * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // find the most effective dimensions for our calculations
    int blockDim = sqrt(deviceProp.maxThreadsPerBlock);
    auto blockSize = dim3(blockDim, blockDim);
    auto gridSize = dim3(round((blockDim - 1 + resopsperiter) / blockDim), round((blockDim - 1 + nAtoms) / blockDim));

    // Launch a kernel on the GPU.
    eFieldDensityGQKernel << <gridSize, blockSize >> > (dev_out, dev_xspans, dev_atom, dev_absci, efp, variance, offset, resopsperiter, nAtoms, resolution);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "density kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching density kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nAtoms * resopsperiter * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(xspans, dev_xspans, nAtoms * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // clear all our device arrays
Error:
    cudaFree(dev_absci);
    cudaFree(dev_xspans);
    cudaFree(dev_atom);
    cudaFree(dev_EFP);
    cudaFree(dev_out);

    return cudaStatus;
}

cudaError_t gaussQuadIntegrationCuda(float *out, const float *inXSpans, const float *inY, const float *inWeights, const size_t nStrips,
    const size_t nPoints, cudaDeviceProp &deviceProp)
{
    // define device arrays
    float *dev_inXSpans = 0;
    float *dev_inWeights = 0;
    float *dev_inY = 0;
    float *dev_out = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nStrips * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inWeights, nPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inXSpans, nStrips * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inY, nStrips * nPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_inXSpans, inXSpans, nStrips * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inWeights, inWeights, nPoints * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inY, inY, nStrips * nPoints * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // use div because it's more accurrate than the rounding BS
    auto gridDiv = div(nStrips, deviceProp.maxThreadsPerBlock);
    auto gridY = gridDiv.quot;

    // ass backwards way of rounding up (maybe use the same trick as above? It might be "faster")
    if (gridDiv.rem != 0)
        gridY++;

    // find the block and grid size
    auto blockSize = deviceProp.maxThreadsPerBlock;
    int gridSize = min(16 * deviceProp.multiProcessorCount, gridY);

    // Launch a kernel on the GPU.
    gaussQuadIntegrationKernel << <gridSize, blockSize >> > (dev_out, dev_inXSpans, dev_inY, dev_inWeights, nStrips, nPoints);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "trapezoid integration kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching trapezoid integration kernel!" << endl;
        cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " '" << cudaGetErrorString(cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nStrips * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    // delete all our device arrays
Error:
    cudaFree(dev_inWeights);
    cudaFree(dev_inXSpans);
    cudaFree(dev_inY);
    cudaFree(dev_out);

    return cudaStatus;
}