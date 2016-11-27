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

#include "ProteinDisplay.h"
#include "PDBProcessor.h"
#include "CSVReader.h"
#include "GPUTypes.h"
#include "gif.h"
#include "Heatmap.h"
#include "helper_string.h"

using namespace std;

#define PI_F 3.14159265358979f;

#define NUM_SLICES 10
#define IMG_SIZE 300

#define IMG_SIZE_SQ (IMG_SIZE * IMG_SIZE)
#define GIF_DELAY 10

// empty definitions that are defined at the bottom later
template<typename T>
cudaError_t sliceDensityCuda(float *out, const T *inAtoms, const GridPoint *inGrid,
    const float variance, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp);

cudaError_t sliceDielectricCuda(float *out, const float *in, const float refDielectric,
    const float outdielectric, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp);

int electricFieldCalculation(string pdbPath, const float lineresolution, const float inDielectric, const float outDielectric, const float variance);

// the density kernel that is ran on the GPU
template<typename T>
__global__ void sliceDensityKernel(float *out, const T *inAtoms, const GridPoint *inGrid,
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

// entry point
int main(int argc, char **argv)
{
    char *pdbFilePath = 0;

    // print the header
    cout << "PFNMR  Copyright(C) 2016 Jonathan Ellis and Bryan Gantt" << endl << endl;
    cout << "This program comes with ABSOLUTELY NO WARRANTY or guarantee of result accuracy." << endl;
    cout << "    This is free software, and you are welcome to redistribute it" << endl;
    cout << "    under certain conditions; see LICENSE for details." << endl << endl;

    // check if we have any input or if "help" or "h" was one of them
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h") ||
        argc < 2)
    {
        cout << "Usage: " << argv[0] << " -file=pdbFile (Required)" << endl;
        cout << "      -oD=OutDielectric (Optional, Default 84.0)" << endl;
        cout << "      -iD=InternalDielectric (Optional, Default 4.0)" << endl;
        cout << "      -rV=RelativeVarience (Optional, Default 0.93)" << endl << endl;

        return 0;
    }

    // check if "file" was defined
    if (checkCmdLineFlag(argc, (const char**)argv, "file"))
    {
        getCmdLineArgumentString(argc, (const char**)argv, "file", &pdbFilePath);
    }
    else
    {
        cout << "The \"-file=pdbFile\" flag is required to run an analysis." << endl;
        cout << "Run \"" << argv[0] << " -help\" for more details." << endl;

        return 1;
    }

    // define some default constants
    auto outDielectric = 84.0f;
    auto inDielectric = 4.0f;
    auto relVariance = 0.93f;

    // check for the optional parameters
    if (checkCmdLineFlag(argc, (const char**)argv, "oD"))
    {
        outDielectric = getCmdLineArgumentFloat(argc, (const char**)argv, "oD");

        if (outDielectric < 0.0f)
        {
            cout << "Error: Value for Out Dielectric must be positive." << endl;
            cout << "Exiting..." << endl;

            return 1;
        }
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "iD"))
    {
        inDielectric = getCmdLineArgumentFloat(argc, (const char**)argv, "iD");

        if (inDielectric < 0.0f)
        {
            cout << "Error: Value for Internal Dielectric must be positive." << endl;
            cout << "Exiting..." << endl;

            return 1;
        }
    }

    // check that ref is less than out
    if (inDielectric >= outDielectric)
    {
        cout << "Error: Internal Dielectric must be less than Out Dielectric." << endl;
        cout << "Exiting..." << endl;

        return 1;
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "rV"))
    {
        relVariance = getCmdLineArgumentFloat(argc, (const char**)argv, "rV");

        if (relVariance < 0.0f)
        {
            cout << "Error: Value for Relative Varience must be positive." << endl;
            cout << "Exiting..." << endl;

            return 1;
        }
    }

    //JME: Launchpoint for things I'm testing.  Done so I don't have to muck about in the main body of code below. 
    //JME: PLEASE NOTE-The code implemented so far has ONLY been tested on an IFABP PDB file with the PHE residues replace for p-fluoro-phenylalanine.  It is potentially very breakable code for other PDBs.
    if (checkCmdLineFlag(argc, (const char**)argv, "test"))
    {
        ProteinDisplay display;
        display.initDisplay();
        return 1;
        //return electricFieldCalculation(pdbFilePath, 1000.0f, inDielectric, outDielectric, relVariance);
    }

    // start a timer for benchmarking
    clock_t startTime = clock();

    // read in a PDB file
    PDBProcessor pdbProcessor(pdbFilePath);

    if (!pdbProcessor.is_open())
    {
        cout << "Failed to open " << pdbFilePath << ". Make sure the file exists or is accessible." << endl << "Exiting..." << endl;
    }

    auto gpuAtoms = pdbProcessor.getGPUAtoms();

    // get the count
    auto nAtoms = gpuAtoms.size();

    // set default pdbBounds
    float pdbBounds[6] = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };

    if (nAtoms != 0)
    {
        // find the bounds
        for (size_t i = 0; i < nAtoms; ++i)
        {
            if (gpuAtoms[i].x < pdbBounds[0])
                pdbBounds[0] = gpuAtoms[i].x;
            else if (gpuAtoms[i].x > pdbBounds[1])
                pdbBounds[1] = gpuAtoms[i].x;

            if (gpuAtoms[i].y < pdbBounds[2])
                pdbBounds[2] = gpuAtoms[i].y;
            else if (gpuAtoms[i].y > pdbBounds[3])
                pdbBounds[3] = gpuAtoms[i].y;

            if (gpuAtoms[i].z < pdbBounds[4])
                pdbBounds[4] = gpuAtoms[i].z;
            else if (gpuAtoms[i].z > pdbBounds[5])
                pdbBounds[5] = gpuAtoms[i].z;
        }

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
            cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
            goto noCuda;
        }

        // find out how much we can calculate
        cudaDeviceProp deviceProp;
        cudaError_t cudaResult;
        cudaResult = cudaGetDeviceProperties(&deviceProp, 0);

        if (cudaResult != cudaSuccess)
        {
            cerr << "cudaGetDeviceProperties failed!" << endl;
            goto noCuda;
        }

        // get how much mem we (in theory) have
        size_t cudaFreeMem;
        cudaResult = cudaMemGetInfo(&cudaFreeMem, NULL);

        if (cudaResult != cudaSuccess)
        {
            cerr << "cudaMemGetInfo failed!" << endl;
            goto noCuda;
        }

        // this nonsense calculates how much we can do at a time for 45% of the memory
        size_t nGpuGridPointBase = floor((cudaFreeMem * 0.45f - (nAtoms * sizeof(GPUAtom))) / ((nAtoms * sizeof(float)) + sizeof(GridPoint)));
        int itersReq = round(IMG_SIZE_SQ / nGpuGridPointBase + 0.5f); // pull some computer math bs to make this work
        auto gridPoints = new GridPoint[IMG_SIZE_SQ];

        // start a new gif writer
        GifWriter gifWriter;

        gifOutputPath.append(pdbFilePath);
        gifOutputPath.resize(gifOutputPath.length() - 4); // strip the .pdb
        gifOutputPath.append(".gif");
        replace(gifOutputPath.begin(), gifOutputPath.end(), ' ', '_'); // replace any spaces that might be in the file

        GifBegin(&gifWriter, gifOutputPath.c_str(), IMG_SIZE, IMG_SIZE, GIF_DELAY);

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

                auto gpuAtomsArray = &gpuAtoms[0];

                // get all the densities for each pixel
                cudaResult = sliceDensityCuda(densityOut, gpuAtomsArray, gpuGridPoints, relVariance, nAtoms, nGpuGridPoint, deviceProp);
                if (cudaResult != cudaSuccess)
                {
                    cout << "Failed to run density kernel." << endl;
                    goto KernelError;
                }

                // get the dielectrics
                cudaResult = sliceDielectricCuda(dielectricOut, densityOut, inDielectric, outDielectric, nAtoms, nGpuGridPoint, deviceProp);
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
                    auto percent = (outDielectric - gridPoints[pixel].dielectric) / (outDielectric - inDielectric);
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

// set up the gpu for the density calculations
template<typename T>
cudaError_t sliceDensityCuda(float *out, const T *inAtoms, const GridPoint *inGrid,
    const float variance, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp)
{
    // define device arrays
    T *dev_atom = 0;
    GridPoint *dev_grid = 0;
    float *dev_out = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for vectors.
    cudaStatus = cudaMalloc((void**)&dev_out, nAtoms * nGridPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_atom, nAtoms * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_grid, nGridPoints * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_atom, inAtoms, nAtoms * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_grid, inGrid, nGridPoints * sizeof(GridPoint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
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
                        phi += PI_F;
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