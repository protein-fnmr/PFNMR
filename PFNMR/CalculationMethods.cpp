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
#include <ctime>
#include <fstream>
#include <math.h>

#include "kernel.cuh"

#include "CalculationMethods.h"
#include "CSVReader.h"
#include "PDBProcessor.h"

using namespace std;

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

int electricFieldCalculation(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance)
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

        //Print back the results
        {
            ofstream logfile("testout.log", ofstream::out);
            cout << "Calculation results:" << endl;
            cout << "ChainId:\tResId:\tField-X:\tField-Y:\tField-Z:\tTotal:\tg09 Input:" << endl;
            logfile << "Calculation results:" << endl;
            logfile << "ChainId:\tResId:\tField-X:\tField-Y:\tField-Z:\tTotal:\tg09 Input:" << endl;

            for (int i = 0; i < nFluorines; i++)
            {
                cout << fluorines[i].chainid << "\t" << fluorines[i].resid << "\t" << fluorines[i].fieldx << "\t" << fluorines[i].fieldy << "\t" << fluorines[i].fieldz << "\t" << fluorines[i].getTotalField() << "\t" << (fluorines[i].getTotalField() * 10000.0f) << "\t" << endl;
                logfile << fluorines[i].chainid << "\t" << fluorines[i].resid << "\t" << fluorines[i].fieldx << "\t" << fluorines[i].fieldy << "\t" << fluorines[i].fieldz << "\t" << fluorines[i].getTotalField() << "\t" << (fluorines[i].getTotalField() * 10000.0f) << "\t" << endl;
            }
            // output the time took
            cout << "Took " << ((clock() - startTime) / ((double)CLOCKS_PER_SEC)) << endl;
            logfile << "Took " << ((clock() - startTime) / ((double)CLOCKS_PER_SEC)) << endl;
            logfile.close();
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
    cout << "Press enter to exit." << endl;
    cin.get();
    return 0;
}
