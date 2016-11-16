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

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <locale>
#include <tuple>
#include <ctime>

#include "GPUTypes.h"
#include "gif.h"
#include "Heatmap.h"
#include "helper_string.h"

using namespace std;

// run a benchmark for the program
// comment out for normal use
//#define BENCHMARK

#ifdef BENCHMARK

#define NUM_SLICES 100
#define IMG_SIZE 500

#else

#define NUM_SLICES 10
#define IMG_SIZE 300

#endif // BENCHMARK

#define IMG_SIZE_SQ (IMG_SIZE * IMG_SIZE)
#define GIF_DELAY 10

// trimming stuff for when we read in a file
string & ltrim(string & str)
{
    auto it2 = find_if(str.begin(), str.end(), [](char ch) { return !isspace<char>(ch, locale::classic()); });
    str.erase(str.begin(), it2);
    return str;
}

string & rtrim(string & str)
{
    auto it1 = find_if(str.rbegin(), str.rend(), [](char ch) { return !isspace<char>(ch, locale::classic()); });
    str.erase(it1.base(), str.end());
    return str;
}

string & trim(string & str)
{
    return ltrim(rtrim(str));
}

// empty definitions that are defined at the bottom later
cudaError_t sliceDensityCuda(float *out, const GPUAtom *inAtoms, const GridPoint *inGrid,
    const float variance, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp);

cudaError_t sliceDielectricCuda(float *out, const float *in, const float refDielectric,
    const float outDialectric, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp);

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
    const float outDialectric, const size_t nAtoms, const size_t nGridPoints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float moldensity = 1.0f;

    if (i < nGridPoints)
    {
        for (int j = 0; j < nAtoms; ++j)
            moldensity *= inDensity[(j * nGridPoints) + i];

        out[i] = ((1.0f - moldensity) * refDielectric) + (moldensity * outDialectric);
    }
}

// entry point
int main(int argc, char **argv)
{
    char *pdbFilePath = 0;

    // print the header
    printf("PFNMR  Copyright(C) 2016 Jonathan Ellis and Bryan Gantt\n\n");
    printf("This program comes with ABSOLUTELY NO WARRANTY or guarantee of result accuracy.\n");
    printf("    This is free software, and you are welcome to redistribute it\n");
    printf("    under certain conditions; see LICENSE for details.\n\n");

    // check if we hav any input or if "help" or "h" was one of them
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h") ||
        argc < 2)
    {
        printf("Usage: %s -file=pdbFile (Required)\n", argv[0]);
        printf("      -oD=OutDielectric (Optional, Default 4.0)\n");
        printf("      -rD=ReferenceDielectric (Optional, Default 80.4)\n");
        printf("      -rV=RelativeVarience (Optional, Default 0.93)\n");

        return 0;
    }

    // check if "file" was defined
    if (checkCmdLineFlag(argc, (const char**)argv, "file"))
    {
        getCmdLineArgumentString(argc, (const char**)argv, "file", &pdbFilePath);
    }
    else
    {
        printf("The \"-file=pdbFile\" flag is required to run an analysis.\n");
        printf("Run \"%s -help\" for more details.\n", argv[0]);

        return 0;
    }

    // start a timer for benchmarking
    clock_t startTime = clock();

    // read in the file
#ifdef BENCHMARK
    ifstream inPdbFile("GroEL.pdb");
#else
    ifstream inPdbFile(pdbFilePath);
#endif // BENCHMARK

    // create a vector
    vector<Atom> atoms;

    // check if we could even open the file
    if (inPdbFile.is_open())
    {
        string line;

        // read each line
        while (getline(inPdbFile, line))
        {
            // read the first 4 characters
            auto begin = line.substr(0, 4);
            if (begin == "ATOM" || begin == "HETA")
            {
                // make an atom and get all the stuff for it
                Atom curAtom;

                // check the element first to see if we
                // need to keep going or not
                auto element = trim(line.substr(76, 2));

                // default vdw is -1.0f, so only check if we need to change it
                // if it's not in the list, just break out (saves a lot of time)
                if (element == "H")
                    curAtom.vdw = 1.2f;
                else if (element == "ZN")
                    curAtom.vdw = 1.39f;
                else if (element == "F")
                    curAtom.vdw = 1.47f;
                else if (element == "O")
                    curAtom.vdw = 1.52f;
                else if (element == "N")
                    curAtom.vdw = 1.55f;
                else if (element == "C")
                    curAtom.vdw = 1.7f;
                else if (element == "S")
                    curAtom.vdw = 1.8f;
                else
                    continue;

                curAtom.element = element;

                auto name = line.substr(12, 4);
                auto resName = line.substr(17, 3);
                auto charge = line.substr(78, 2);

                curAtom.serial = stoi(line.substr(6, 5));
                curAtom.name = trim(name);
                curAtom.altLoc = line.at(16);
                curAtom.resName = trim(resName);
                curAtom.chainID = line.at(21);
                curAtom.resSeq = stoi(line.substr(22, 4));
                curAtom.iCode = line.at(26);
                curAtom.x = stof(line.substr(30, 8));
                curAtom.y = stof(line.substr(38, 8));
                curAtom.z = stof(line.substr(46, 8));
                curAtom.occupancy = stof(line.substr(54, 6));
                curAtom.tempFactor = stof(line.substr(60, 6));
                curAtom.charge = trim(charge);

                // if we have a valid vdw, add it to the vector
                if (curAtom.vdw != -1.0f)
                    atoms.push_back(curAtom);
            }
        }

        cout << "Found " << atoms.size() << " atoms.\n";

        // close the file
        inPdbFile.close();
    }
    else
    {
        cout << "Unable to open " << pdbFilePath << "\n";
        return 1;
    }

    // get the count
    auto nAtoms = atoms.size();

    // set default pdbBounds
    float pdbBounds[6] = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };

    if (nAtoms != 0)
    {
        // find the bounds
        for (size_t i = 0; i < nAtoms; ++i)
        {
            if (atoms[i].x < pdbBounds[0])
                pdbBounds[0] = atoms[i].x;
            else if (atoms[i].x > pdbBounds[1])
                pdbBounds[1] = atoms[i].x;

            if (atoms[i].y < pdbBounds[2])
                pdbBounds[2] = atoms[i].y;
            else if (atoms[i].y > pdbBounds[3])
                pdbBounds[3] = atoms[i].y;

            if (atoms[i].z < pdbBounds[4])
                pdbBounds[4] = atoms[i].z;
            else if (atoms[i].z > pdbBounds[5])
                pdbBounds[5] = atoms[i].z;
        }

        // define some constants
        const auto outDielectric = 84.0f;
        const auto refDielectric = 4.0f;
        const auto relVariance = 0.93f;

        // this shouldn't be an issue though because we know that size > 0
        auto gpuAtoms = new GPUAtom[nAtoms];
        
        // copy the atoms over
        for (size_t i = 0; i < nAtoms; ++i)
            gpuAtoms[i] = { atoms[i].x, atoms[i].y, atoms[i].z, atoms[i].vdw };

        atoms.clear(); // we shouldn't need these anymore (or really at all?)

        // find the spans
        auto xspan = pdbBounds[1] - pdbBounds[0];
        auto yspan = pdbBounds[3] - pdbBounds[2];
        auto zspan = pdbBounds[5] - pdbBounds[4];

        // find the center of the bounds
        float boxCenter[3];
        boxCenter[0] = (xspan / 2) + pdbBounds[0];
        boxCenter[1] = (yspan / 2) + pdbBounds[2];
        boxCenter[2] = (zspan / 2) + pdbBounds[4];

        // expand the bounds for a border
        xspan *= 1.1f;
        yspan *= 1.1f;
        zspan *= 1.1f;

        // this is doing X. Purely for benchmark purposes
        // logic needs to be added later for oter directions
        string gifOutputPath = "x_slice_";
        auto maxSpan = max(yspan, zspan);
        auto pointStep = maxSpan / (IMG_SIZE - 1);

        // move the view to the new location
        pdbBounds[0] = boxCenter[0] - (xspan / 2);
        pdbBounds[1] = boxCenter[0] + (xspan / 2);
        pdbBounds[2] = boxCenter[1] - (maxSpan / 2);
        pdbBounds[3] = boxCenter[1] + (maxSpan / 2);
        pdbBounds[4] = boxCenter[2] - (maxSpan / 2);
        pdbBounds[5] = boxCenter[2] + (maxSpan / 2);

        // THIS WILL EVENTUALLY BE LOOPED FOR MULTI-GPU
        // Choose which GPU to run on, change this on a multi-GPU system.
        if (cudaSetDevice(0) != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
            goto noCuda;
        }

        // find out how much we can calculate
        cudaDeviceProp deviceProp;
        cudaError_t cudaResult;
        cudaResult = cudaGetDeviceProperties(&deviceProp, 0);

        if (cudaResult != cudaSuccess)
        {
            fprintf(stderr, "cudaGetDeviceProperties failed!\n");
            goto noCuda;
        }

        // get how much mem we (in theory) have
        size_t cudaFreeMem;
        cudaResult = cudaMemGetInfo(&cudaFreeMem, NULL);

        if (cudaResult != cudaSuccess)
        {
            fprintf(stderr, "cudaMemGetInfo failed!\n");
            goto noCuda;
        }

        // this nonsense calculates how much we can do at a time for 45% of the memory
        size_t nGpuGridPointBase = floor((cudaFreeMem * 0.45f - (nAtoms * sizeof(GPUAtom))) / ((nAtoms * sizeof(float)) + sizeof(GridPoint)));
        int itersReq = round(IMG_SIZE_SQ / nGpuGridPointBase + 0.5f); // pull some computer math bs to make this work
        auto gridPoints = new GridPoint[IMG_SIZE_SQ];

        // start a new gif writer
        GifWriter gifWriter;

#ifdef BENCHMARK
        GifBegin(&gifWriter, "x_slice_groel.gif", IMG_SIZE, IMG_SIZE, GIF_DELAY);
#else
        gifOutputPath.append(pdbFilePath);
        gifOutputPath.resize(gifOutputPath.length() - 4); // strip the .pdb
        gifOutputPath.append(".gif");
        replace(gifOutputPath.begin(), gifOutputPath.end(), ' ', '_'); // replace any spaces that might be in the file

        GifBegin(&gifWriter, gifOutputPath.c_str(), IMG_SIZE, IMG_SIZE, GIF_DELAY);
#endif // BENCHMARK

        // do NUM_SLICES slices
        for (int slice = 0; slice < NUM_SLICES; ++slice)
        {
            cout << "Slice " << slice + 1 << " of " << NUM_SLICES << endl;

            // THIS RIGHT HERE WAS THE PROBLEM THE WHOLE TIME
            // force a float value with 1.0f bs
            auto xval = ((slice + 1.0f) / (NUM_SLICES + 1)) * xspan + pdbBounds[0];

            // input all the new points into the grid
            for (int y = 0; y < IMG_SIZE; ++y)
            {
                auto yval = pdbBounds[2] + (y * pointStep);

                for (int z = 0; z < IMG_SIZE; ++z)
                {
                    size_t loc = (y * IMG_SIZE) + z;
                    gridPoints[loc].x = xval;
                    gridPoints[loc].y = yval;
                    gridPoints[loc].z = pdbBounds[4] + (z * pointStep);
                }
            }

            // go over every chunk (either every slice goes to the GPU, or every subslice...)
            for (int subslice = 0; subslice < itersReq; ++subslice)
            {
                // start with the base number of points
                auto nGpuGridPoint = nGpuGridPointBase;

                // if we're at the end, we just use what is needed
                if ((IMG_SIZE_SQ - nGpuGridPointBase * subslice) < nGpuGridPointBase)
                    nGpuGridPoint = IMG_SIZE_SQ - nGpuGridPoint * subslice;

                // create the gridpoint subset array
                auto gpuGridPoints = new GridPoint[nGpuGridPoint];

                // push values over to the gridpoint subset array
                for (size_t j = 0; j < nGpuGridPoint; ++j)
                    gpuGridPoints[j] = gridPoints[j + subslice * nGpuGridPointBase];

                // create new arrays to store the output
                auto densityOut = new float[nAtoms * nGpuGridPoint];
                auto dielectricOut = new float[nGpuGridPoint];

                // get all the densities for each pixel
                cudaResult = sliceDensityCuda(densityOut, gpuAtoms, gpuGridPoints, relVariance, nAtoms, nGpuGridPoint, deviceProp);
                if (cudaResult != cudaSuccess)
                {
                    cout << "Failed to run density kernel." << endl;
                    goto KernelError;
                }

                // get the dielectrics
                cudaResult = sliceDielectricCuda(dielectricOut, densityOut, refDielectric, outDielectric, nAtoms, nGpuGridPoint, deviceProp);
                if (cudaResult != cudaSuccess)
                {
                    cout << "Failed to run dielectric kernel." << endl;
                    goto KernelError;
                }

                // copy the dielectric values from the gpu return back to the main gridpoint array
                for (size_t j = 0; j < nGpuGridPoint; ++j)
                    gridPoints[j + subslice * nGpuGridPointBase].dielectric = dielectricOut[j];

                // delete all the stuff we don't need anymore
            KernelError:
                delete[] dielectricOut;
                delete[] densityOut;
                delete[] gpuGridPoints;

                // if we didn't work the first time, don't keep going
                if (cudaResult != cudaSuccess)
                    goto kernelFailed;
            }

            // define a new array to store our image data
            auto image = new uint8_t[IMG_SIZE_SQ * 4];

            // move over each pixel
            for (int y = 0; y < IMG_SIZE; ++y)
            {
                for (int x = 0; x < IMG_SIZE; ++x)
                {
                    // find the pixel location and it's heatmap value
                    auto pixel = (y * IMG_SIZE) + x;
                    auto percent = (outDielectric - gridPoints[pixel].dielectric) / (outDielectric - refDielectric);
                    auto heat = getHeatMapColor(percent);

                    // a pixel has 4 colors so mult by 4 and offset for each color
                    // 0: R, 1: G, 2: B, 3: A
                    image[pixel * 4] = (uint8_t)floor(heat[0] * 255);
                    image[pixel * 4 + 1] = (uint8_t)floor(heat[1] * 255);
                    image[pixel * 4 + 2] = (uint8_t)floor(heat[2] * 255);
                    image[pixel * 4 + 3] = 255;

                    delete[] heat;
                }
            }

            // write out a frame
            GifWriteFrame(&gifWriter, image, IMG_SIZE, IMG_SIZE, GIF_DELAY);
            delete[] image;
        }

        // use this to break out of the nested for loop
        // using more conditionals is "better" but not as efficient
        // close the gif because it was open, even if no frames were written
    kernelFailed:
        GifEnd(&gifWriter);
        delete[] gridPoints;

    noCuda:
        delete[] gpuAtoms;
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
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 3;
    }

    // output the time took
    cout << "Took " << ((clock() - startTime) / ((double)CLOCKS_PER_SEC)) << endl;

    return 0;
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

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nAtoms * nGridPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_atom, nAtoms * sizeof(GPUAtom));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_grid, nGridPoints * sizeof(GridPoint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_atom, inAtoms, nAtoms * sizeof(GPUAtom), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_grid, inGrid, nGridPoints * sizeof(GridPoint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // find the most effective dimensions for our calculations
    int blockDim = sqrt(deviceProp.maxThreadsPerBlock);
    auto blockSize = dim3(blockDim, blockDim);
    auto gridSize = dim3(round((blockDim - 1 + nGridPoints) / blockDim), round((blockDim - 1 + nAtoms) / blockDim));

    // Launch a kernel on the GPU.
    sliceDensityKernel <<<gridSize, blockSize>>> (dev_out, dev_atom, dev_grid, variance, nAtoms, nGridPoints);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "density kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching density kernel!\n", cudaStatus);
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nAtoms * nGridPoints * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
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
    const float outDialectric, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp)
{
    // the device arrays
    float *dev_in = 0;
    float *dev_out = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors
    cudaStatus = cudaMalloc((void**)&dev_out, nGridPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_in, nAtoms * nGridPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, in, nAtoms * nGridPoints * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // use div because it's more accurrate than the rounding BS
    auto gridDiv = div(nGridPoints, deviceProp.maxThreadsPerBlock);
    auto gridY = gridDiv.quot;

    // ass backwards way of rounding up (maybe use the same trick as above? It might be "faster")
    if (gridDiv.rem != 0)
        gridY++;

    // find the block and grid size
    auto blockSize = deviceProp.maxThreadsPerBlock;
    int gridSize = min(16 * deviceProp.multiProcessorCount, gridY);

    // Launch a kernel on the GPU.
    sliceDielectricKernel <<<gridSize, blockSize>>> (dev_out, dev_in, refDielectric, outDialectric, nAtoms, nGridPoints);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dielectric launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dielectric!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, nGridPoints * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // delete all our device arrays
Error:
    cudaFree(dev_in);
    cudaFree(dev_out);

    return cudaStatus;
}
