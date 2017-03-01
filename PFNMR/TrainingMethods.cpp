#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>

#include "kernel.cuh"

#include "Heatmap.h"
#include "TrainingMethods.h"
#include "CalculationMethods.h"
#include "CSVReader.h"
#include "PDBProcessor.h"
#include "PFDProcessor.h"


void electricFieldCalculationSilent(cudaDeviceProp deviceProp, vector<Atom> & baseatoms, vector<GPUEFP> & fluorines, vector<GPUChargeAtom> & gpuatoms, vector<float> & weights, vector<float> & abscissa, const float inDielectric, const float outDielectric, const float variance, vector<float> & output)
{
    int actres = abscissa.size();
    int nAtoms = baseatoms.size();
    int nFluorines = fluorines.size();

    auto gpuatomsarray = &gpuatoms[0];
    auto gpuabscissa = &abscissa[0];
    auto gpuweights = &weights[0];
    auto gpuefparray = &fluorines[0];

    //Make sure we can use the graphics card (This calculation would be unresonable otherwise)
    if (cudaSetDevice(0) != cudaSuccess) {
        cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
        goto noCuda;
    }

    cudaError_t cudaResult;

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
    size_t freemem = (cudaFreeMem * 0.25f) - (nAtoms * sizeof(GPUChargeAtom)) - (nFluorines * sizeof(GPUEFP)) - memdielectricmatrix - (2 * nAtoms * sizeof(float));
    if (freemem < 0)
    {
        cout << "Error: Not enough memory for this calculation!" << endl;
        return;
    }
    int slicesperiter = floor(freemem / memdielectricmatrix);
    int iterReq = round(nAtoms / slicesperiter + 0.5f);
    int resopsperiter = slicesperiter * actres;

    //Start doing the actual analysis using the Effective Length method for the non-uniform dielectric
    auto integralMatrix = new float[nAtoms * nFluorines];
    auto alpha = (((2.99792458f * 2.99792458f) * 1.602176487f) / (5.142206f)) * 0.1f; //Conversion factor to make calculation spit out voltage in Gaussian 09 atomic units. Analogous to Coulomb's Constant.
    for (int i = 0; i < fluorines.size(); i++)  //Cycle through each fluorine
    {
        auto dielectricMatrix = new float[nAtoms * actres]; //The end dielectric matrix that will be used to calculate the field components
        auto xspans = new float[nAtoms];
        for (int j = 0; j < iterReq; j++) //Keep going until we process all the iteration chunks
        {
            auto densityMatrix = new float[nAtoms * resopsperiter]; //Temporary density matrix
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
        output.resize(fluorinatedAAs.size());
        for (int i = 0; i < fluorinatedAAs.size(); i++)
        {
            vector<float> fieldVect{ fluorines[i].fieldx, fluorines[i].fieldy, fluorines[i].fieldz };
            rotateResidueToXField(fieldVect, fluorinatedAAs[i]);

            //Do NMR calculation
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
            auto field = fluorines[i].getTotalField() * 5000.0f;
            auto nmr = pheNMR(anglex, angley, anglez, 80.4f, field);
            output[i] = nmr;

        }
    }
kernelFailed:;

noCuda:;
    delete[] integralMatrix;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    auto cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceReset failed!" << endl;
        return;
    }
    return;
}