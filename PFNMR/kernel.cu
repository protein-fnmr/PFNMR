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
