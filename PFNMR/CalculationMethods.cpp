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

#define NUM_SPHERE_POINTS   10000
#define SPHERE_RADIUS       1.0f

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>

#include "kernel.cuh"

#include "Heatmap.h"
#include "CalculationMethods.h"
#include "CSVReader.h"
#include "PDBProcessor.h"
#include "PFDProcessor.h"

using namespace std;

void rotateResidueToXField(vector<float> & fieldVect, vector<Atom> residue);

vector<float> crossprod(vector<float> & a, vector<float> & b)
{
    vector<float> result;
    result.push_back((a[1] * b[2]) - (a[2] * b[1]));
    result.push_back((a[2] * b[0]) - (a[0] * b[2]));
    result.push_back((a[0] * b[1]) - (a[1] * b[0]));
    return result;
}

float pheNMR(float x, float y, float z, float d, float w)
{
    //Average parameters from Monte Carlo fitting
    //auto a = 0.0f;  //-116.58f;
    //auto b = 0.009077721f;
    //auto c = 0.101656497f;
    //auto d = 0.01746456f;

    //Old method without dielectric information
    //return a + (b*field) - (c * field * cosf((d * y) - (d*z))) - (c * field * cosf((d * z) + (d*y)));

    //Super specific method (with dielectric)
    return (0.00478f * w) + ((5.8f + (0.0457f * w * cosf(0.0175f * y) * cosf(0.0175f * z))) / d) + (0.0457f * w * cosf(0.0175f * z) * cosf(0.0175f * z) * sinf(0.00478f * w * cosf(0.0175f * y))) - (0.203f * w * cosf(0.0175f * y) * cosf(0.0175f * z));
}

void getGaussQuadSetup(int points, vector<float> & outWeights, vector<float> & outAbscissa)
{
    switch (points)
    {
    case 20:
        outWeights.push_back(0.1527533871307258f);
        outWeights.push_back(0.1527533871307258f);
        outWeights.push_back(0.1491729864726037f);
        outWeights.push_back(0.1491729864726037f);
        outWeights.push_back(0.1420961093183820f);
        outWeights.push_back(0.1420961093183820f);
        outWeights.push_back(0.1316886384491766f);
        outWeights.push_back(0.1316886384491766f);
        outWeights.push_back(0.1181945319615184f);
        outWeights.push_back(0.1181945319615184f);
        outWeights.push_back(0.1019301198172404f);
        outWeights.push_back(0.1019301198172404f);
        outWeights.push_back(0.0832767415767048f);
        outWeights.push_back(0.0832767415767048f);
        outWeights.push_back(0.0626720483341091f);
        outWeights.push_back(0.0626720483341091f);
        outWeights.push_back(0.0406014298003869f);
        outWeights.push_back(0.0406014298003869f);
        outWeights.push_back(0.0176140071391521f);
        outWeights.push_back(0.0176140071391521f);
        outAbscissa.push_back(-0.0765265211334973f);
        outAbscissa.push_back(0.0765265211334973f);
        outAbscissa.push_back(-0.2277858511416451f);
        outAbscissa.push_back(0.2277858511416451f);
        outAbscissa.push_back(-0.3737060887154195f);
        outAbscissa.push_back(0.3737060887154195f);
        outAbscissa.push_back(-0.5108670019508271f);
        outAbscissa.push_back(0.5108670019508271f);
        outAbscissa.push_back(-0.6360536807265150f);
        outAbscissa.push_back(0.6360536807265150f);
        outAbscissa.push_back(-0.7463319064601508f);
        outAbscissa.push_back(0.7463319064601508f);
        outAbscissa.push_back(-0.8391169718222188f);
        outAbscissa.push_back(0.8391169718222188f);
        outAbscissa.push_back(-0.9122344282513259f);
        outAbscissa.push_back(0.9122344282513259f);
        outAbscissa.push_back(-0.9639719272779138f);
        outAbscissa.push_back(0.9639719272779138f);
        outAbscissa.push_back(-0.9931285991850949f);
        outAbscissa.push_back(0.9931285991850949f);
        break;
    case 10:
    default:
        outWeights.push_back(0.2955242247147529f);
        outWeights.push_back(0.2955242247147529f);
        outWeights.push_back(0.2692667193099963f);
        outWeights.push_back(0.2692667193099963f);
        outWeights.push_back(0.2190863625159820f);
        outWeights.push_back(0.2190863625159820f);
        outWeights.push_back(0.1494513491505806f);
        outWeights.push_back(0.1494513491505806f);
        outWeights.push_back(0.0666713443086881f);
        outWeights.push_back(0.0666713443086881f);
        outAbscissa.push_back(-0.1488743389816312f);
        outAbscissa.push_back(0.1488743389816312f);
        outAbscissa.push_back(-0.4333953941292472f);
        outAbscissa.push_back(0.4333953941292472f);
        outAbscissa.push_back(-0.6794095682990244f);
        outAbscissa.push_back(0.6794095682990244f);
        outAbscissa.push_back(-0.8650633666889845f);
        outAbscissa.push_back(0.8650633666889845f);
        outAbscissa.push_back(-0.9739065285171717f);
        outAbscissa.push_back(0.9739065285171717f);
    }
}

int createDielectricPFDFile(string outpfdpath, string pdbFilePath, string colorcsvpath, int nSlices, int imgSize, float outDielectric, float inDielectric, float relVariance)
{
    int imgSizeSq = imgSize * imgSize;
    PDBProcessor pdbProcessor(pdbFilePath);

    if (!pdbProcessor.is_open())
    {
        cout << "Failed to open " << pdbFilePath << ". Make sure the file exists or is accessible." << endl << "Exiting..." << endl;
        return 1;
    }
    auto atoms = pdbProcessor.getAtomsFromPDB();
    auto gpuAtoms = pdbProcessor.getGPUAtomsFromAtoms(atoms);

    CSVReader csv(colorcsvpath);
    auto colortable = csv.readCSVFile();

    PFDWriter pfd;
    openPFDFileWriter(&pfd, outpfdpath);
    writeStructurePFDInfo(&pfd, atoms, colortable);

    // get the count
    auto nAtoms = gpuAtoms.size();

    // set default pdbBounds
    float pdbBounds[6] = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };

    if (nAtoms != 0)
    {
        vector<float> hmrange;
        hmrange.push_back(inDielectric);
        hmrange.push_back(outDielectric);
        writeHeatmapSetHeader(&pfd, nSlices, imgSize, hmrange);

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
        size_t nGpuGridPointBase = floor((cudaFreeMem * 0.25f - (nAtoms * sizeof(GPUAtom))) / ((nAtoms * sizeof(float)) + sizeof(GridPoint)));
        int itersReq = round(imgSizeSq / nGpuGridPointBase + 0.5f); // pull some computer math bs to make this work
        auto gridPoints = new GridPoint[imgSizeSq];

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

            //Write image data to file
            
            vector<float> planedims;
            planedims.push_back(xval);
            planedims.push_back(pdbBounds[2]);
            planedims.push_back(pdbBounds[4]);
            planedims.push_back(xval);
            planedims.push_back(pdbBounds[3]);
            planedims.push_back(pdbBounds[5]);

            
            auto image = new float[imgSizeSq];
            for (int i = 0; i < imgSizeSq; i++)
            {
                image[i] = gridPoints[i].dielectric;
            }
            writeHeatmapFrameData(&pfd, image, planedims, imgSize);
            
            /*
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
            writeDielectricFrameData(&pfd, image, planedims, imgSize);
            */
            delete[] image;
        }

    kernelFailed:
        delete[] gridPoints;
        closePFDFileWriter(&pfd);

    noCuda:;
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
}

int oldElectricFieldCalculation(string pdbPath, const float lineresolution, const float inDielectric, const float outDielectric, const float variance)
{
    clock_t startTime = clock();
    //Read the charge table and get the appropriate charged atoms
    CSVReader csvreader("ChargeTable.csv");
    auto chargetable = csvreader.readCSVFile();
    PDBProcessor pdb(pdbPath);
    auto baseatoms = pdb.getAtomsFromPDB();
    auto nAtoms = baseatoms.size();
    auto gpuatoms = pdb.getGPUChargeAtomsFromAtoms(baseatoms, chargetable);
    auto gpuatomsarray = &gpuatoms[0];

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

        auto nFluorines = fluorines.size();

        if (nFluorines == 0)
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
        for (int i = 0; i < nFluorines; i++)  //Cycle through each fluorine
        {
            cout << "Processing fluorine " << (i + 1) << "/" << nFluorines << endl;
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
                    auto theta = asinf(diffy / distance);
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
        for (int i = 0; i < nFluorines; i++)
        {
            cout << fluorines[i].chainid << "\t" << fluorines[i].resid << "\t" << fluorines[i].fieldx << "\t" << fluorines[i].fieldy << "\t" << fluorines[i].fieldz << "\t" << fluorines[i].getTotalField() << "\t" << (fluorines[i].getTotalField() * 10000.0f) << "\t" << endl;
        }

    kernelFailed:

    noCuda:;

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
    cout << "Press enter to exit." << endl;
    cin.get();
    return 0;
}

int electricFieldCalculation(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance, vector<float> & output)
{
    clock_t startTime = clock();
    //Read the charge table and get the appropriate charged atoms
    CSVReader csvreader("ChargeTable.csv");
    auto chargetable = csvreader.readCSVFile();
    PDBProcessor pdb(pdbPath);
    auto baseatoms = pdb.getAtomsFromPDB();
    auto nAtoms = baseatoms.size();
    auto gpuatoms = pdb.getGPUChargeAtomsFromAtoms(baseatoms, chargetable);
    auto plaingpuatoms = pdb.getGPUAtomsFromAtoms(baseatoms);
    auto gpuatomsarray = &gpuatoms[0];

    vector<float> weights;
    vector<float> abscissa;
    getGaussQuadSetup(res, weights, abscissa);
    int actres = abscissa.size();
    cout << "Using " << actres << " points for integration!" << endl;
    auto gpuabscissa = &abscissa[0];
    auto gpuweights = &weights[0];


    if (nAtoms != 0)
    {
        //Find all the fluorines that will be processed
        vector<GPUEFP> fluorines;
        vector<int> fluorinesindicies;
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
                fluorinesindicies.push_back(i);
            }
        }
        auto gpuefparray = &fluorines[0];
        auto nFluorines = fluorines.size();
        if (nFluorines == 0)
        {
            cout << "Error: There are no fluorines in the PDB provided." << endl;
            return 1;
        }

        //Make sure we can use the graphics card (This calculation would be unresonable otherwise)
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

        // Calculate available memory
        auto memdielectricmatrix = (actres * nAtoms * sizeof(float));
        size_t freemem = (cudaFreeMem * 0.10f) - (nAtoms * sizeof(GPUChargeAtom)) - (nFluorines * sizeof(GPUEFP)) - memdielectricmatrix - (2 * nAtoms * sizeof(float));
        if (freemem < 0)
        {
            cout << "Error: Not enough memory for this calculation!" << endl;
            return 1;
        }
        int slicesperiter = floor(freemem / memdielectricmatrix);
        int iterReq = round(nAtoms / slicesperiter + 0.5f);
        int resopsperiter = slicesperiter * actres;

        //Start doing the actual analysis using the Effective Length method for the non-uniform dielectric
        auto integralMatrix = new float[nAtoms * nFluorines];
        auto alpha = (((2.99792458f * 2.99792458f) * 1.602176487f) / (5.142206f)) * 0.1f; //Conversion factor to make calculation spit out voltage in Gaussian 09 atomic units. Analogous to Coulomb's Constant.
        cout << "Beginning electric field calculations using \"effective length\" treatment for non-uniform dielectric." << endl;
        for (int i = 0; i < fluorines.size(); i++)  //Cycle through each fluorine
        {
            cout << "Processing fluorine " << (i + 1) << "/" << fluorines.size() << endl;
            auto dielectricMatrix = new float[nAtoms * actres]; //The end dielectric matrix that will be used to calculate the field components
            auto xspans = new float[nAtoms];
            for (int j = 0; j < iterReq; j++) //Keep going until we process all the iteration chunks
            {
                auto densityMatrix = new float[nAtoms * resopsperiter]; //Temporary density matrix
                cout << "\t Iteration: " << (j + 1) << "/" << iterReq << endl;
                //Run the density kernel to populate the density matrix
                eFieldDensityGQCuda(densityMatrix, xspans, gpuatomsarray, gpuabscissa, fluorines[i], variance, j * resopsperiter, resopsperiter, nAtoms, actres, deviceProp);
                if (cudaResult != cudaSuccess)
                {
                    cout << "Failed to run density kernel." << endl;
                    goto InnerKernelError;
                }

                //Run the dielectric kernel on the density matrix, and store the results in the dielectric matrix
                eFieldDielectricCuda(dielectricMatrix, densityMatrix, inDielectric, outDielectric, j * resopsperiter, resopsperiter, nAtoms, actres, deviceProp);
                if (cudaResult != cudaSuccess)
                {
                    cout << "Failed to run dielectric kernel." << endl;
                    goto InnerKernelError;
                }
            InnerKernelError:
                delete[] densityMatrix;
                // if we didn't work the first time, don't keep going
                if (cudaResult != cudaSuccess)
                    goto kernelFailed;
            }

            //Sqrt all the dielectrics (Needed for effective length method)
            sqrtf2DCuda(dielectricMatrix, nAtoms, actres, deviceProp);
            if (cudaResult != cudaSuccess)
            {
                cout << "Failed to run sqrtf 2D kernel." << endl;
                goto KernelError;
            }

            //Do the integration to get the effective length
            //trapIntegrationCuda(&integralMatrix[i * nAtoms], xspans, dielectricMatrix, nAtoms, actres, deviceProp);
            gaussQuadIntegrationCuda(&integralMatrix[i*nAtoms], xspans, dielectricMatrix, gpuweights, nAtoms, actres, deviceProp);
            if (cudaResult != cudaSuccess)
            {
                cout << "Failed to run trapezoid integration kernel." << endl;
                goto KernelError;
            }

        KernelError:
            delete[] xspans;
            delete[] dielectricMatrix;

            if (cudaResult != cudaSuccess)
                goto kernelFailed;
        }

        //Calculate all the field components and store them in the EFP matrix
        electricFieldComponentCuda(gpuefparray, integralMatrix, gpuatomsarray, alpha, nFluorines, nAtoms, deviceProp);
        if (cudaResult != cudaSuccess)
        {
            cout << "Failed to run electric field component kernel." << endl;
            goto kernelFailed;
        }

        //Print back the electric field results
        {
#ifdef OUTPUT_LOG
            ofstream logfile("testout.log", ofstream::out);
            cout << "Calculation results:" << endl;
            cout << "ChainId:\tResId:\tField-X:\tField-Y:\tField-Z:\tTotal:\tg09 Input:" << endl;
            logfile << "Calculation results:" << endl;
            logfile << "ChainId:\tResId:\tField-X:\tField-Y:\tField-Z:\tTotal:\tg09 Input:" << endl;
#endif
            for (int i = 0; i < nFluorines; i++)
            {
                cout << fluorines[i].chainid << "\t" << fluorines[i].resid << "\t" << fluorines[i].fieldx << "\t" << fluorines[i].fieldy << "\t" << fluorines[i].fieldz << "\t" << fluorines[i].getTotalField() << "\t" << (fluorines[i].getTotalField() * 10000.0f) << "\t" << endl;
#ifdef OUTPUT_LOG
                logfile << fluorines[i].chainid << "\t" << fluorines[i].resid << "\t" << fluorines[i].fieldx << "\t" << fluorines[i].fieldy << "\t" << fluorines[i].fieldz << "\t" << fluorines[i].getTotalField() << "\t" << (fluorines[i].getTotalField() * 10000.0f) << "\t" << endl;
#endif
            }           

            //Get all the geometries of the fluorinated amino acids
            vector<vector<Atom>> fluorinatedAAs;
            for (int i = 0; i < fluorines.size(); i++)
            {
                vector<Atom> temp;
                for (int j = 0; j < baseatoms.size(); j++)
                {
                    if ((baseatoms[j].resSeq == fluorines[i].resid))
                    {
                        temp.push_back(baseatoms[j]);
                    }
                }
                fluorinatedAAs.push_back(temp);
            }

            //Rotate all the residues by the field vectors
            cout << "Rotating residues to align electric field to x-axis..." << endl;
#ifdef OUTPUT_LOG
            logfile << "Rotating residues to align electric field to x-axis..." << endl;
            logfile << "Element:\tx:\ty:\tz:" << endl;
#endif
            for (int i = 0; i < fluorinatedAAs.size(); i++)
            {
#ifdef OUTPUT_LOG
                logfile << endl << "Chain: " << fluorinatedAAs[i][0].chainID << "\tResID:" << fluorinatedAAs[i][0].resSeq << endl;
 #endif
                vector<float> fieldVect{ fluorines[i].fieldx, fluorines[i].fieldy, fluorines[i].fieldz };
                rotateResidueToXField(fieldVect, fluorinatedAAs[i]);
#ifdef OUTPUT_LOG
                for (int j = 0; j < fluorinatedAAs[i].size(); j++)
                {
                    logfile << fluorinatedAAs[i][j].element << "\t" << fluorinatedAAs[i][j].x << "\t" << fluorinatedAAs[i][j].y << "\t" << fluorinatedAAs[i][j].z << endl;
                }
#endif
            }

            //Start the actual NMR calculation based on the Monte Carlo fit data
            
            cout << endl;
            cout << "Beginning Monte Carlo fit based NMR calculation:" << endl;
#ifdef OUTPUT_LOG
            logfile << endl;
            logfile << "Monte Carlo fit based NMR calculation:" << endl;
#endif
            output.resize(fluorinatedAAs.size());

            for (int i = 0; i < fluorinatedAAs.size(); i++)
            {
                //Build the coordinate template matrix
                //TODO: This ONLY works with Phenylalanine.  Make this more universal with a look-up table for the coordinate indicies.
                vector<vector<float>> coordtemplate;
                coordtemplate.resize(3);
                for (int i = 0; i < 3; i++)
                {
                    coordtemplate[i].resize(3);
                }
                coordtemplate[0][0] = fluorinatedAAs[i][11].x;   coordtemplate[0][1] = fluorinatedAAs[i][10].x;  coordtemplate[0][2] = fluorinatedAAs[i][9].x;
                coordtemplate[1][0] = fluorinatedAAs[i][11].y;   coordtemplate[1][1] = fluorinatedAAs[i][10].y;  coordtemplate[1][2] = fluorinatedAAs[i][9].y;
                coordtemplate[2][0] = fluorinatedAAs[i][11].z;   coordtemplate[2][1] = fluorinatedAAs[i][10].z;  coordtemplate[2][2] = fluorinatedAAs[i][9].z;

                //Ensure the structure is centered properly
                //TODO: This might not be necessary.  Double check.
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        coordtemplate[j][i] -= coordtemplate[j][1];
                    }
                }

                //Get the proposed de-rotated structure
                vector<vector<float>> rotationmatrix;
                rotationmatrix.resize(3);
                for (int i = 0; i < 3; i++)
                {
                    rotationmatrix[i].resize(3);
                }
                vector<float> xvect;
                xvect.push_back(coordtemplate[0][0]);
                xvect.push_back(coordtemplate[1][0]);
                xvect.push_back(coordtemplate[2][0]);
                rotationmatrix[0][0] = xvect[0];
                rotationmatrix[1][0] = xvect[1];
                rotationmatrix[2][0] = xvect[2];

                vector<float> ccvect;
                ccvect.push_back(coordtemplate[0][2]);
                ccvect.push_back(coordtemplate[1][2]);
                ccvect.push_back(coordtemplate[2][2]);
                auto zvect = crossprod(xvect, ccvect);
                rotationmatrix[0][2] = zvect[0];
                rotationmatrix[1][2] = zvect[1];
                rotationmatrix[2][2] = zvect[2];

                auto yvect = crossprod(zvect, xvect);
                rotationmatrix[0][1] = yvect[0];
                rotationmatrix[1][1] = yvect[1];
                rotationmatrix[2][1] = yvect[2];

                //Ensure rotation matrix is a unit vector matrix
                for (int j = 0; j < 3; j++)
                {
                    auto mag = 0.0f;
                    for (int k = 0; k < 3; k++)
                    {
                        mag += (rotationmatrix[k][j] * rotationmatrix[k][j]);
                    }
                    mag = sqrtf(mag);
                    for (int k = 0; k < 3; k++)
                    {
                        rotationmatrix[k][j] /= mag;
                    }
                }

                //Solve for the angles
                auto angley = asinf(-1.0f * rotationmatrix[2][0]);
                auto anglex = asinf(rotationmatrix[2][1] / cosf(angley));
                auto anglez = asinf(rotationmatrix[1][0] / cosf(angley));
                anglex *= (360.0f / (2.0f * M_PI));
                angley *= (360.0f / (2.0f * M_PI));
                anglez *= (360.0f / (2.0f * M_PI));


                //Get the NMR shift from the Monte Carlo fit equation
                auto field = fluorines[i].getTotalField() * 10000.0f;
                auto dielectric = calculateAverageDielectric(10000, 1.0f, plaingpuatoms, plaingpuatoms[fluorinesindicies[i]], variance, inDielectric, outDielectric);
                auto nmr = pheNMR(anglex, angley, anglez, dielectric, field);
                output[i] = nmr;
                cout << "Residue: " << fluorinatedAAs[i][0].resSeq << "\t(" << anglex << "," << angley << "," << anglez << "," << dielectric << "," << field << "):\t" << nmr << endl;
#ifdef OUTPUT_LOG
                logfile << "Residue: " << fluorinatedAAs[i][0].resSeq << "\t(" << anglex << "," << angley << "," << anglez << "," << dielectric << "," << field << "):\t" << nmr << endl;
#endif
                output[i] = nmr;
            }
            // output the time took
            cout << "Took " << ((clock() - startTime) / ((double)CLOCKS_PER_SEC)) << endl;
#ifdef OUTPUT_LOG
            logfile << "Took " << ((clock() - startTime) / ((double)CLOCKS_PER_SEC)) << endl << endl;
            logfile.close();
#endif
        }
    kernelFailed:;

    noCuda:;
        delete[] integralMatrix;
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
    return 0;
}

int electricPotentialCalculation(string pdbPath, const int integralres, const int nSlices, const int gridres, const float inDielectric, const float outDielectric, const float variance)
{
    clock_t startTime = clock();
    //Read the charge table and get the appropriate charged atoms
    CSVReader csvreader("ChargeTable.csv");
    auto chargetable = csvreader.readCSVFile();
    PDBProcessor pdb(pdbPath);
    auto baseatoms = pdb.getAtomsFromPDB();
    auto nAtoms = baseatoms.size();
    auto gpuatoms = pdb.getGPUChargeAtomsFromAtoms(baseatoms, chargetable);
    auto gpuatomsarray = &gpuatoms[0];

    auto gridlen = gridres * gridres;

    vector<float> weights;
    vector<float> abscissa;
    getGaussQuadSetup(integralres, weights, abscissa);
    int actres = abscissa.size();
    cout << "Using " << actres << " points for integration!" << endl;
    auto gpuabscissa = &abscissa[0];
    auto gpuweights = &weights[0];

    float pdbBounds[6] = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };

    if (nAtoms != 0)
    {
        for (size_t i = 0; i < nAtoms; ++i)
        {
            if (gpuatoms[i].x < pdbBounds[0])
                pdbBounds[0] = gpuatoms[i].x;
            else if (gpuatoms[i].x > pdbBounds[1])
                pdbBounds[1] = gpuatoms[i].x;

            if (gpuatoms[i].y < pdbBounds[2])
                pdbBounds[2] = gpuatoms[i].y;
            else if (gpuatoms[i].y > pdbBounds[3])
                pdbBounds[3] = gpuatoms[i].y;

            if (gpuatoms[i].z < pdbBounds[4])
                pdbBounds[4] = gpuatoms[i].z;
            else if (gpuatoms[i].z > pdbBounds[5])
                pdbBounds[5] = gpuatoms[i].z;
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
        auto maxSpan = max(yspan, zspan);
        auto pointStep = maxSpan / (gridres - 1);

        // move the view to the new location
        pdbBounds[0] = boxCenter[0] - (xspan / 2);
        pdbBounds[1] = boxCenter[0] + (xspan / 2);
        pdbBounds[2] = boxCenter[1] - (maxSpan / 2);
        pdbBounds[3] = boxCenter[1] + (maxSpan / 2);
        pdbBounds[4] = boxCenter[2] - (maxSpan / 2);
        pdbBounds[5] = boxCenter[2] + (maxSpan / 2);

        //Make sure we can use the graphics card (This calculation would be unresonable otherwise)
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

        // Calculate available memory
        auto memdielectricmatrix = (actres * nAtoms * sizeof(float));
        size_t freemem = (cudaFreeMem * 0.10f) - (nAtoms * sizeof(GPUChargeAtom)) - sizeof(GPUEFP) - memdielectricmatrix - (2 * nAtoms * sizeof(float));
        if (freemem < 0)
        {
            cout << "Error: Not enough memory for this calculation!" << endl;
            return 1;
        }
        int slicesperiter = floor(freemem / memdielectricmatrix);
        int iterReq = round(nAtoms / slicesperiter + 0.5f);
        int resopsperiter = slicesperiter * actres;

        //Start doing the actual analysis using the Effective Length method for the non-uniform dielectric
        auto integralMatrix = new float[nAtoms];
        auto alpha = (((2.99792458f * 2.99792458f) * 1.602176487f) / (5.142206f)) * 0.1f; //Conversion factor to make calculation spit out voltage in Gaussian 09 atomic units. Analogous to Coulomb's Constant.
        cout << "Beginning electric potential calculations using \"effective length\" treatment for non-uniform dielectric." << endl;
        float* results = new float[nSlices * gridlen];
        for (int slice = 0; slice < nSlices; slice++)  //Cycle through each fluorine
        {
            cout << "Processing slice " << slice + 1 << " of " << nSlices << endl;
            auto xval = ((slice + 1.0f) / (nSlices + 1)) * xspan + pdbBounds[0];

            for (int fieldpoint = 0; fieldpoint < gridlen; fieldpoint++)
            {
                cout << "Processing step " << fieldpoint + 1 << " of " << gridlen << "\r";
                GPUEFP testpoint;
                testpoint.x = xval;
                testpoint.y = fieldpoint % gridres;
                testpoint.z = fieldpoint / gridres;

                auto dielectricMatrix = new float[nAtoms * actres]; //The end dielectric matrix that will be used to calculate the field components
                auto xspans = new float[nAtoms];
                for (int j = 0; j < iterReq; j++) //Keep going until we process all the iteration chunks
                {
                    auto densityMatrix = new float[nAtoms * resopsperiter]; //Temporary density matrix
                    //Run the density kernel to populate the density matrix
                    eFieldDensityGQCuda(densityMatrix, xspans, gpuatomsarray, gpuabscissa, testpoint, variance, j * resopsperiter, resopsperiter, nAtoms, actres, deviceProp);
                    if (cudaResult != cudaSuccess)
                    {
                        cout << "Failed to run density kernel." << endl;
                        goto InnerKernelError;
                    }

                    //Run the dielectric kernel on the density matrix, and store the results in the dielectric matrix
                    eFieldDielectricCuda(dielectricMatrix, densityMatrix, inDielectric, outDielectric, j * resopsperiter, resopsperiter, nAtoms, actres, deviceProp);
                    if (cudaResult != cudaSuccess)
                    {
                        cout << "Failed to run dielectric kernel." << endl;
                        goto InnerKernelError;
                    }
                InnerKernelError:
                    delete[] densityMatrix;
                    // if we didn't work the first time, don't keep going
                    if (cudaResult != cudaSuccess)
                        goto kernelFailed;
                }

                //Sqrt all the dielectrics (Needed for effective length method)
                sqrtf2DCuda(dielectricMatrix, nAtoms, actres, deviceProp);
                if (cudaResult != cudaSuccess)
                {
                    cout << "Failed to run sqrtf 2D kernel." << endl;
                    goto KernelError;
                }

                //Do the integration to get the effective length
                //trapIntegrationCuda(&integralMatrix[i * nAtoms], xspans, dielectricMatrix, nAtoms, actres, deviceProp);
                gaussQuadIntegrationCuda(integralMatrix, xspans, dielectricMatrix, gpuweights, nAtoms, actres, deviceProp);
                if (cudaResult != cudaSuccess)
                {
                    cout << "Failed to run trapezoid integration kernel." << endl;
                    goto KernelError;
                }
                //Calculate all the field components and store them in the EFP matrix
                electricPotentialCuda(&results[(slice * gridlen) + fieldpoint], integralMatrix, gpuatomsarray, alpha, 1, nAtoms, deviceProp);
                if (cudaResult != cudaSuccess)
                {
                    cout << "Failed to run electric field component kernel." << endl;
                    goto kernelFailed;
                }
            KernelError:
                delete[] xspans;
                delete[] dielectricMatrix;

                if (cudaResult != cudaSuccess)
                    goto kernelFailed;
            }

        }


        //Print back the results
        cout << "Took " << ((clock() - startTime) / ((double)CLOCKS_PER_SEC)) << endl;
    kernelFailed:;

    noCuda:;
        delete[] integralMatrix;
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
    cout << "Press enter to exit." << endl;
    cin.get();
    return 0;
}

void rotateResidueToXField(vector<float> & fieldVect, vector<Atom> residue)
{
    //Ensure the field vector is a unit vector
    auto fieldMag = sqrtf((fieldVect[0] * fieldVect[0]) + (fieldVect[1] * fieldVect[1]) + (fieldVect[2] * fieldVect[2]));
    fieldVect[0] /= fieldMag;
    fieldVect[1] /= fieldMag;
    fieldVect[2] /= fieldMag;

    //Calculate the rotation matrix based on: http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    auto scalar = (1 - fieldVect[0]) / ((fieldVect[2] * fieldVect[2]) + (fieldVect[1] * fieldVect[1]));
    vector<vector<float>> rotmatrix
    {
        { 1.0f, fieldVect[1], fieldVect[2] },
        { -fieldVect[1], 1.0f, 0.0f },
        { -fieldVect[2], 0.0f, 1.0f }
    };
    vector<vector<float>> rightmatrix
    {
        { (0.0f - (fieldVect[1] * fieldVect[1]) - (fieldVect[2] * fieldVect[2])) * scalar, 0.0f, 0.0f},
        { 0.0f, (0.0f - (fieldVect[1] * fieldVect[1])) * scalar, (0.0f - (fieldVect[1] * fieldVect[2])) * scalar},
        { 0.0f, (0.0f - (fieldVect[1] * fieldVect[2])) * scalar, (0.0f - (fieldVect[2] * fieldVect[2])) * scalar }
    };
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            rotmatrix[i][j] += rightmatrix[i][j];
        }
    }

    //Apply the rotation matrix to the residue to spin it around the origin
    //TODO: Potentially have to convert coordinates to a unit vector set based on the fluorine atom position (its magnitude as the divisor)
    vector<float> moleculedims{ FLT_MAX, FLT_MIN, FLT_MAX, FLT_MIN, FLT_MAX, FLT_MIN };
    for (int i = 0; i < residue.size(); i++)
    {
        vector<float> newcoords
        {
            (rotmatrix[0][0] * residue[i].x) + (rotmatrix[0][1] * residue[i].y) + (rotmatrix[0][2] * residue[i].z),
            (rotmatrix[1][0] * residue[i].x) + (rotmatrix[1][1] * residue[i].y) + (rotmatrix[1][2] * residue[i].z),
            (rotmatrix[2][0] * residue[i].x) + (rotmatrix[2][1] * residue[i].y) + (rotmatrix[2][2] * residue[i].z)
        };
        residue[i].x = newcoords[0];
        residue[i].y = newcoords[1];
        residue[i].z = newcoords[2];

        if (residue[i].x < moleculedims[0])
        {
            moleculedims[0] = residue[i].x;
        }
        if (residue[i].x > moleculedims[1])
        {
            moleculedims[1] = residue[i].x;
        }
        if (residue[i].y < moleculedims[2])
        {
            moleculedims[2] = residue[i].y;
        }
        if (residue[i].y > moleculedims[3])
        {
            moleculedims[3] = residue[i].y;
        }
        if (residue[i].z < moleculedims[4])
        {
            moleculedims[4] = residue[i].z;
        }
        if (residue[i].z > moleculedims[5])
        {
            moleculedims[5] = residue[i].z;
        }
    }

    //Center the geometry
    vector<float> centerpoint{ (moleculedims[0] + moleculedims[1]) / 2.0f, (moleculedims[2] + moleculedims[3]) / 2.0f, (moleculedims[4] + moleculedims[5]) / 2.0f };
    for (int i = 0; i < residue.size(); i++)
    {
        residue[i].x -= centerpoint[0];
        residue[i].y -= centerpoint[1];
        residue[i].z -= centerpoint[2];
    }
}

float calculateAverageDielectric(int numpoints, float sphererad, vector<GPUAtom> atoms, GPUAtom & target, float variance, float inDielectric, float outDielectric)
{
    //Setup a dummy sphere grid using cube carving
    float average = -1.0f;
    vector<GridPoint> spherepoints;
    auto griddim = (int)floorf(cbrtf(numpoints) / 2.0f);
    auto dimstep = sphererad / (float)griddim;
    for (int x = 0; x < griddim; x++)
    {
        for (int y = 0; y < griddim; y++)
        {
            for (int z = 0; z < griddim; z++)
            {
                auto dist = sqrtf(((x * dimstep) * (x * dimstep)) + ((y * dimstep) * (y * dimstep)) + ((z * dimstep) * (z * dimstep)));
                if (dist < sphererad && (x + y + z) != 0)
                {
                    GridPoint temp;
                    temp.x = x * dimstep;
                    temp.y = y * dimstep;
                    temp.z = z * dimstep;
                    spherepoints.push_back(temp);
                    temp.x *= -1.0f;
                    spherepoints.push_back(temp);
                    temp.y *= -1.0f;
                    spherepoints.push_back(temp);
                    temp.z *= -1.0f;
                    spherepoints.push_back(temp);
                    temp.x *= -1.0f;
                    spherepoints.push_back(temp);
                    temp.y *= -1.0f;
                    spherepoints.push_back(temp);
                    temp.x *= -1.0f;
                    spherepoints.push_back(temp);
                    temp.x *= -1.0f;
                    temp.y *= -1.0f;
                    temp.z *= -1.0f;
                    spherepoints.push_back(temp);
                }
            }
        }
    }

    //Move the sphere points
    for (int i = 0; i < spherepoints.size(); i++)
    {
        spherepoints[i].x += target.x;
        spherepoints[i].y += target.y;
        spherepoints[i].z += target.z;
    }

    cudaDeviceProp deviceProp;
    cudaError_t cudaResult;
    cudaResult = cudaGetDeviceProperties(&deviceProp, 0);

    if (cudaResult != cudaSuccess)
    {
        cerr << "cudaGetDeviceProperties failed!" << endl;
        goto noCUDA;
    }

    auto densitygrid = new float[atoms.size() * spherepoints.size()];
    auto gpuatoms = &atoms[0];
    auto gpugrid = &spherepoints[0];
    auto outdigrid = new float[spherepoints.size()];

    sliceDensityCudaIR(densitygrid, gpuatoms, gpugrid, variance, target.resid, atoms.size(), spherepoints.size(), deviceProp);
    if (cudaResult != cudaSuccess)
    {
        cout << "Failed to run dielectric kernel." << endl;
        goto InnerKernelError;
    }
    sliceDielectricCuda(outdigrid, densitygrid, inDielectric, outDielectric, atoms.size(), spherepoints.size(), deviceProp);
    if (cudaResult != cudaSuccess)
    {
        cout << "Failed to run dielectric kernel." << endl;
        goto InnerKernelError;
    }
    average = 0.0f;
    for (int i = 0; i < spherepoints.size(); i++)
    {
        average += outdigrid[i];
    }
    average /= (float)spherepoints.size();

InnerKernelError:
    delete[] densitygrid;
    delete[] outdigrid;
noCUDA:
    return average;
}