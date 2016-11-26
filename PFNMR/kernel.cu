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
//template<typename T>
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
//template<typename T>
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

#if 0
int electricFieldCalculation(string pdbPath, const float lineresolution, const float inDielectric, const float outDielectric, const float variance)
{
    clock_t startTime = clock();
    //Read the charge table and get the appropriate charged atoms
    CSVReader csvreader("ChargeTable.csv");
    auto chargetable = csvreader.readCSVFile();
    PDBProcessor pdb(pdbPath);
    auto gpuatoms = pdb.getGPUChargeAtoms(chargetable);
    auto gpuatomsarray = &gpuatoms[0];
    pdb.restart();
    auto baseatoms = pdb.getAtomsFromPDB();
    auto nAtoms = gpuatoms.size();

    if (nAtoms != 0)
    {
        //Find all the fluorines that will be processed
        vector<GPUEFP> fluorines;
        for (int i = 0; i < baseatoms.size(); i++)
        {
            if (baseatoms[i].element == "F")
            {
                GPUEFP newefp;
                newefp.x = baseatoms[i].x;
                newefp.y = baseatoms[i].y;
                newefp.z = baseatoms[i].z;
                newefp.chainid = (int)baseatoms[i].chainID;
                newefp.resid = baseatoms[i].resSeq;
                fluorines.push_back(newefp);
            }
        }

        if (fluorines.size() == 0)
        {
            cout << "Error: There are no fluorines in the PDB provided." << endl;
            return 1;
        }

        //Make sure we can use the graphics card (This calculation would be unresonable otherwise)
        if (cudaSetDevice(0) != cudaSuccess) {
            cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
            goto noCuda;
        }

        cudaDeviceProp deviceProp;
        cudaError_t cudaResult;
        cudaResult = cudaGetDeviceProperties(&deviceProp, 0);

        if (cudaResult != cudaSuccess)
        {
            cerr << "cudaGetDeviceProperties failed!" << endl;
            goto noCuda;
        }

        //Start doing the actual analysis using the Effective Length method for the non-uniform dielectric
        //NOTE: I know this is an absolutely terrible/inefficient way of doing this, but this framework will be used to devize a better method (Already in the works.)
        auto alpha = (((2.99792458f * 2.99792458f) * 1.602176487f) / (5.142206f)) * 0.1f; //Conversion factor to make calculation spit out voltage in Gaussian 09 atomic units. Analogous to Coulomb's Constant.
        cout << "Beginning electric field calculations using \"effective length\" treatment for non-uniform dielectric." << endl;
        for (int i = 0; i < fluorines.size(); i++)  //Cycle through each fluorine
        {
            cout << "Processing fluorine " << (i+1) << "/" << fluorines.size() << endl;
            for (int j = 0; j < nAtoms; j++) //Get the field contribution of each atom
            {
                if (fluorines[i].resid != gpuatoms[j].resid || fluorines[i].chainid != gpuatoms[j].chainid) //Make sure we dont process atoms on the same residue, since these will be handled using QM methods.
                {
                    //Get all the parameters for putting gridpoints along  a line between the fluorine and atom in question, as well as info for the electric field calcuation.
                    auto diffx = fluorines[i].x - gpuatoms[j].x;
                    auto diffy = fluorines[i].y - gpuatoms[j].y;
                    auto diffz = fluorines[i].z - gpuatoms[j].z;
                    if (diffx == 0.0) //Catch if this is zero, since this would break the electric field calculation later.
                        diffx = FLT_EPSILON;
                    auto distance = sqrtf((diffx * diffx) + (diffy * diffy) + (diffz * diffz));

                    //Get the spacing between each point along the line
                    auto xstep = diffx / (lineresolution + 1.0f);
                    auto ystep = diffy / (lineresolution + 1.0f);
                    auto zstep = diffz / (lineresolution + 1.0f);

                    //Populate a GridPoint array with the points on the line
                    auto gpulinepoints = new GridPoint[(int)lineresolution];
                    for (int k = 1; k <= lineresolution; k++)
                    {
                        gpulinepoints[k - 1].x = gpuatoms[j].x + (xstep * k);
                        gpulinepoints[k - 1].y = gpuatoms[j].y + (ystep * k);
                        gpulinepoints[k - 1].z = gpuatoms[j].z + (zstep * k);
                    }

                    //Do the density and dielectric calculations for the line
                    auto densityOut = new float[nAtoms * (int)lineresolution];
                    auto dielectricOut = new float[(int)lineresolution];
                    cudaResult = sliceDensityCuda(densityOut, gpuatomsarray, gpulinepoints, variance, nAtoms, (int)lineresolution, deviceProp);
                    if (cudaResult != cudaSuccess)
                    {
                        cout << "Failed to run density kernel." << endl;
                        goto KernelError;
                    }
                    cudaResult = sliceDielectricCuda(dielectricOut, densityOut, inDielectric, outDielectric, nAtoms, (int)lineresolution, deviceProp);
                    if (cudaResult != cudaSuccess)
                    {
                        cout << "Failed to run dielectric kernel." << endl;
                        goto KernelError;
                    }

                    //Apply sqrtf to all dielectrics, and calculate the effective length using the trapezoid integral method.
                    auto pointspacing = distance / (lineresolution + 1.0f);
                    auto effectivelength = 0.0f;
                    for (int k = 1; k < lineresolution; k++)
                    {
                        effectivelength += (pointspacing * ((sqrtf(dielectricOut[k]) + sqrtf(dielectricOut[k - 1])) / 2.0f));
                    }
                    //Time to actually calculate the electric field components
                    auto Etot = ((gpuatoms[j].charge * alpha) / (effectivelength * effectivelength));
                    auto theta = asinf(diffy / effectivelength);
                    auto phi = atanf(diffz / diffx);
                    if (diffx < 0.0f)
                        phi += M_PI;
                    fluorines[i].fieldx += Etot * cosf(theta) * cosf(phi);
                    fluorines[i].fieldy += Etot * sinf(theta);
                    fluorines[i].fieldz += Etot * cosf(theta) * sinf(phi);

                KernelError:
                    delete[] dielectricOut;
                    delete[] densityOut;
                    delete[] gpulinepoints;

                    // if we didn't work the first time, don't keep going
                    if (cudaResult != cudaSuccess)
                        goto kernelFailed;
                }
            }
            cout << fluorines[i].chainid << "\t" << fluorines[i].resid << "\t" << fluorines[i].fieldx << "\t" << fluorines[i].fieldy << "\t" << fluorines[i].fieldz << "\t" << fluorines[i].getTotalField() << "\t" << (fluorines[i].getTotalField() * 10000.0f) << "\t" << endl;
        }

        //Print back the results
        cout << "Calculation results:" << endl;
        cout << "ChainId:\tResId:\tField-X:\tField-Y:\tField-Z:\tTotal:\tg09 Input:" << endl;
        for (int i = 0; i < fluorines.size(); i++)
        {
            cout << fluorines[i].chainid << "\t" << fluorines[i].resid << "\t" << fluorines[i].fieldx << "\t" << fluorines[i].fieldy << "\t" << fluorines[i].fieldz << "\t" << fluorines[i].getTotalField() << "\t" << (fluorines[i].getTotalField() * 10000.0f) << "\t" << endl;
        }


    kernelFailed:

    noCuda:

    }
    else
    {
        cout << "Found no valid atoms. Exiting..." << endl;
        return 2;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    auto cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceReset failed!" << endl;
        return 3;
    }

    // output the time took
    cout << "Took " << ((clock() - startTime) / ((double)CLOCKS_PER_SEC)) << endl;

    return 0;
}
#endif
