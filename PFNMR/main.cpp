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

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <math.h>

#include "kernel.cuh"
#include "CalculationMethods.h"
#include "PDBProcessor.h"
#include "CSVReader.h"
#include "GPUTypes.h"
#include "Heatmap.h"
#include "gif.h"
#include "helper_string.h"

#include "GaussQuadrature.h"

using namespace std;

#define EFIELDTESTING

#define GIF_DELAY 10

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
        cout << "      -oD=OutDielectric (Optional, Default 80.4)" << endl;
        cout << "      -iD=InternalDielectric (Optional, Default 4.0)" << endl;
        cout << "      -rV=RelativeVarience (Optional, Default 0.93)" << endl << endl;
		cout << "      -slices=SliceCount (Optional, Default 10)" << endl;
		cout << "      -imgSize=ImageSize (Optional, Default 300)" << endl << endl;

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
    auto outDielectric = 80.4f;
    auto inDielectric = 4.0f;
    auto relVariance = 0.93f;

	auto nSlices = 10;
	auto imgSize = 300;
	auto imgSizeSq = 90000;

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

	// check for the slices override
	if (checkCmdLineFlag(argc, (const char**)argv, "slices"))
	{
		nSlices = getCmdLineArgumentInt(argc, (const char**)argv, "slices");

		if (nSlices <= 0)
		{
			cout << "Warning: the Slice Count must be greater than zero." << endl;
			cout << "Reverting to default value: 10" << endl;

			nSlices = 10;
		}
	}

	// check for image size override
	// TODO: this can get nasty of someone puts in a number that's larger than sqrt(INT_MAX)
	// but who really wants to make a gif that's 46341px x 46341px?
	if (checkCmdLineFlag(argc, (const char**)argv, "imgSize"))
	{
		imgSize = getCmdLineArgumentInt(argc, (const char**)argv, "imgSize");

		if (imgSize <= 0)
		{
			cout << "Warning: the Image Size must be greater than zero." << endl;
			cout << "Reverting to default value: 300" << endl;

			imgSize = 300;
		}

		imgSizeSq = imgSize * imgSize;
	}

#ifdef EFIELDTESTING
    //JME: Launchpoint for things I'm testing.  Done so I don't have to muck about in the main body of code below. 
    //JME: PLEASE NOTE-The code implemented so far has ONLY been tested on an IFABP PDB file with the PHE residues replace for p-fluoro-phenylalanine.  It is potentially very breakable code for other PDBs.
    if (checkCmdLineFlag(argc, (const char**)argv, "test"))
    {
        return electricFieldCalculation(pdbFilePath, 10, inDielectric, outDielectric, relVariance);
    }
#endif

    // start a timer for benchmarking
    clock_t startTime = clock();

    // read in a PDB file
    PDBProcessor pdbProcessor(pdbFilePath);

    if (!pdbProcessor.is_open())
    {
        cout << "Failed to open " << pdbFilePath << ". Make sure the file exists or is accessible." << endl << "Exiting..." << endl;
        return 1;
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
        auto pointStep = maxSpan / (imgSize - 1);

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
        int itersReq = round(imgSizeSq / nGpuGridPointBase + 0.5f); // pull some computer math bs to make this work
        auto gridPoints = new GridPoint[imgSizeSq];

        // start a new gif writer
        GifWriter gifWriter;

        gifOutputPath.append(pdbFilePath);
        gifOutputPath.resize(gifOutputPath.length() - 4); // strip the .pdb
        gifOutputPath.append(".gif");
        replace(gifOutputPath.begin(), gifOutputPath.end(), ' ', '_'); // replace any spaces that might be in the file

        GifBegin(&gifWriter, gifOutputPath.c_str(), imgSize, imgSize, GIF_DELAY);

        // perform the operation over every slice
        for (int slice = 0; slice < nSlices; ++slice)
        {
            cout << "Calculating slice " << slice + 1 << " of " << nSlices << endl;

            // THIS RIGHT HERE WAS THE PROBLEM THE WHOLE TIME
            // force a float value with 1.0f bs
            auto xval = ((slice + 1.0f) / (nSlices + 1)) * xspan + pdbBounds[0];

            // input all the new points into the grid
            for (int y = 0; y < imgSize; ++y)
            {
                auto yval = pdbBounds[2] + (y * pointStep);

                for (int z = 0; z < imgSize; ++z)
                {
                    size_t loc = (y * imgSize) + z;
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
                if ((imgSizeSq - nGpuGridPointBase * subslice) < nGpuGridPointBase)
                    nGpuGridPoint = imgSizeSq - nGpuGridPoint * subslice;

                // create the gridpoint subset array
                auto gpuGridPoints = new GridPoint[nGpuGridPoint];

                // push values over to the gridpoint subset array
                for (size_t j = 0; j < nGpuGridPoint; ++j)
                    gpuGridPoints[j] = gridPoints[j + subslice * nGpuGridPointBase];

                // create new arrays to store the output
                auto densityOut = new float[nAtoms * nGpuGridPoint];
                auto dielectricOut = new float[nGpuGridPoint];

                // get all the densities for each pixel
                cudaResult = sliceDensityCuda(densityOut, &gpuAtoms[0], gpuGridPoints, relVariance, nAtoms, nGpuGridPoint, deviceProp);
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
            auto image = new uint8_t[imgSizeSq * 4];

            // move over each pixel
            for (int y = 0; y < imgSize; ++y)
            {
                for (int x = 0; x < imgSize; ++x)
                {
                    // find the pixel location and it's heatmap value
                    auto pixel = (y * imgSize) + x;
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
            GifWriteFrame(&gifWriter, image, imgSize, imgSize, GIF_DELAY);
            delete[] image;
        }

        // use this to break out of the nested for loop
        // using more conditionals is "better" but not as efficient
        // close the gif because it was open, even if no frames were written
    kernelFailed:
        GifEnd(&gifWriter);
        delete[] gridPoints;

    noCuda: ;
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