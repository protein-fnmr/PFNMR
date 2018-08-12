#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>
#include <algorithm>

#include "kernel.cuh"

#include "Heatmap.h"
#include "TrainingMethods.h"
#include "CalculationMethods.h"
#include "CSVReader.h"
#include "PDBProcessor.h"
//#include "PFDProcessor.h"


float errorfunc(float calculated, float correct)
{
	return abs(calculated - correct);
}

float rounderhelper(float x, float step)
{
	return floor((x*(1.0f / step)) + step) / (1.0f / step);
}

int electricFieldCalculationSilent(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance, vector<float> & output)
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
		unsigned long long memdielectricmatrix;
        memdielectricmatrix = (actres * nAtoms * sizeof(float));
		size_t freemem;
        freemem = (cudaFreeMem * 0.10f) - (nAtoms * sizeof(GPUChargeAtom)) - (nFluorines * sizeof(GPUEFP)) - memdielectricmatrix - (2 * nAtoms * sizeof(float));
		if (freemem < 0)
		{
			cout << "Error: Not enough memory for this calculation!" << endl;
			return 1;
		}
		int slicesperiter;
        slicesperiter = floor(freemem / memdielectricmatrix);
		int iterReq;
        iterReq = round(nAtoms / slicesperiter + 0.5f);
		int resopsperiter;
        resopsperiter = slicesperiter * actres;
        
		//Start doing the actual analysis using the Effective Length method for the non-uniform dielectric
		float* integralMatrix;
        integralMatrix = new float[nAtoms * nFluorines];
		float alpha;
        alpha = (((2.99792458f * 2.99792458f) * 1.602176487f) / (5.142206f)) * 0.1f; //Conversion factor to make calculation spit out voltage in Gaussian 09 atomic units. Analogous to Coulomb's Constant.
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

			for (int i = 0; i < fluorinatedAAs.size(); i++)
			{
				vector<float> fieldVect{ fluorines[i].fieldx, fluorines[i].fieldy, fluorines[i].fieldz };
				rotateResidueToXField(fieldVect, fluorinatedAAs[i]);
			}

			//Start the actual NMR calculation based on the Monte Carlo fit data

			output.resize(fluorinatedAAs.size());

			for (int i = 0; i < fluorinatedAAs.size(); i++)
			{

				//Time to construct the geometry vector for the residue
				//Search through and find the fluorine
				int fluorid = -1;
				for (int j = 0; j < fluorinatedAAs[i].size(); ++j)
				{
					if (fluorinatedAAs[i][j].element == "F")
					{
						fluorid = j;
						break;
					}
				}

				if (fluorid == -1)
				{
					cout << "ERROR: Couldn't find fluorine for residue " << fluorinatedAAs[i][0].resSeq << endl;
					continue;
				}

				//Find the carbon its bound to
				auto shortestdist = 10000.0f;
				int boundcarbon = -1;
				for (int j = 0; j < fluorinatedAAs[i].size(); ++j)
				{
					if (fluorinatedAAs[i][j].element == "C")
					{
						auto currdist = sqrtf(((fluorinatedAAs[i][j].x - fluorinatedAAs[i][fluorid].x) * (fluorinatedAAs[i][j].x - fluorinatedAAs[i][fluorid].x)) +
							((fluorinatedAAs[i][j].y - fluorinatedAAs[i][fluorid].y) * (fluorinatedAAs[i][j].y - fluorinatedAAs[i][fluorid].y)) +
							((fluorinatedAAs[i][j].z - fluorinatedAAs[i][fluorid].z) * (fluorinatedAAs[i][j].z - fluorinatedAAs[i][fluorid].z)));
						if (currdist < shortestdist)
						{
							shortestdist = currdist;
							boundcarbon = j;
						}
					}
				}

				if (boundcarbon == -1)
				{
					cout << "ERROR: Couldn't find a carbon to bind to fluorine for residue " << fluorinatedAAs[i][0].resSeq << endl;
					continue;
				}

				//Calculate the geometry vector
				vector<float> geomvect = {
					(fluorinatedAAs[i][fluorid].x - fluorinatedAAs[i][boundcarbon].x),
					(fluorinatedAAs[i][fluorid].y - fluorinatedAAs[i][boundcarbon].y),
					(fluorinatedAAs[i][fluorid].z - fluorinatedAAs[i][boundcarbon].z)
				};

				//Calculate Euler angles
				auto angles = generateGeometryRotationAnglesToX(geomvect);
				angles[0] *= (360.0f / (2.0f * (float)M_PI));
				angles[1] *= (360.0f / (2.0f * (float)M_PI));
				angles[2] *= (360.0f / (2.0f * (float)M_PI));

				//Get the NMR shift from the Monte Carlo fit equation
				auto field = fluorines[i].getTotalField() * 10000.0f;
				auto dielectric = calculateAverageDielectric(10000, 1.0f, plaingpuatoms, plaingpuatoms[fluorinesindicies[i]], variance, inDielectric, outDielectric);
				//For testing purposes, since this is acting funny
				auto nmr = pheNMR(angles[0], angles[1], angles[2], dielectric, field);
				output[i] = nmr;
			}
			// output the time took
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

int electricFieldCalculationSilentReporter(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance, vector<float> & output, vector<float> & outputneg, vector<float> & flipoutput, vector<float> & flipoutputneg, ofstream & report)
{
	string tagline = to_string(inDielectric) + "\t" + to_string(outDielectric) + "\t" + to_string(variance) + "\t";

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
	auto gpuabscissa = &abscissa[0];
	auto gpuweights = &weights[0];


	if (nAtoms != 0)
	{
		//Find all the fluorines that will be processed
		vector<GPUEFP> fluorines;
		vector<int> fluorinesindicies;
		vector<string> reportlines;
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
				reportlines.push_back(tagline + to_string(newefp.resid) + "\t");
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
		unsigned long long memdielectricmatrix;
        memdielectricmatrix = (actres * nAtoms * sizeof(float));
		size_t freemem;
        freemem = (cudaFreeMem * 0.10f) - (nAtoms * sizeof(GPUChargeAtom)) - (nFluorines * sizeof(GPUEFP)) - memdielectricmatrix - (2 * nAtoms * sizeof(float));
		if (freemem < 0)
		{
			cout << "Error: Not enough memory for this calculation!" << endl;
			return 1;
		}
		int slicesperiter;
        slicesperiter = floor(freemem / memdielectricmatrix);
		int iterReq;
        iterReq = round(nAtoms / slicesperiter + 0.5f);
		int resopsperiter;
        resopsperiter = slicesperiter * actres;

		//Start doing the actual analysis using the Effective Length method for the non-uniform dielectric
		float* integralMatrix;
        integralMatrix = new float[nAtoms * nFluorines];
		float alpha;
        alpha = (((2.99792458f * 2.99792458f) * 1.602176487f) / (5.142206f)) * 0.1f; //Conversion factor to make calculation spit out voltage in Gaussian 09 atomic units. Analogous to Coulomb's Constant.
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

		{
			//Get all the geometries of the fluorinated amino acids
			vector<vector<Atom>> fluorinatedAAs;
			vector<vector<Atom>> flipfluorinatedAAs;
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
				flipfluorinatedAAs.push_back(temp);
			}

			for (int i = 0; i < fluorinatedAAs.size(); i++)
			{
				vector<float> fieldVect{ fluorines[i].fieldx, fluorines[i].fieldy, fluorines[i].fieldz };
				auto rotmat = rotateResidueToXField(fieldVect, fluorinatedAAs[i]);
				rotmat[0] *= (360.0f / (2.0f * (float)M_PI));
				rotmat[1] *= (360.0f / (2.0f * (float)M_PI));
				rotmat[2] *= (360.0f / (2.0f * (float)M_PI));
				reportlines[i] += to_string(fluorines[i].fieldx) + "\t" + to_string(fluorines[i].fieldy) + "\t" + to_string(fluorines[i].fieldz) + 
					"\t" + to_string(fluorines[i].getTotalField()) + "\t" + to_string(rotmat[0]) + "\t" + to_string(rotmat[1]) + "\t" + 
					to_string(rotmat[2]) + "\t";

				vector<float> flipfieldVect{ -1.0f * fluorines[i].fieldx, -1.0f * fluorines[i].fieldy, -1.0f * fluorines[i].fieldz };
				auto flipfieldstr = sqrtf((flipfieldVect[0] * flipfieldVect[0]) + (flipfieldVect[1] * flipfieldVect[1]) + (flipfieldVect[2] * flipfieldVect[2]));
				auto fliprotmat = rotateResidueToXField(flipfieldVect, flipfluorinatedAAs[i]);
				fliprotmat[0] *= (360.0f / (2.0f * (float)M_PI));
				fliprotmat[1] *= (360.0f / (2.0f * (float)M_PI));
				fliprotmat[2] *= (360.0f / (2.0f * (float)M_PI));
				reportlines[i] += to_string(flipfieldVect[0]) + "\t" + to_string(flipfieldVect[1]) + "\t" + to_string(flipfieldVect[2]) +
					"\t" + to_string(flipfieldstr) + "\t" + to_string(fliprotmat[0]) + "\t" + to_string(rotmat[1]) + "\t" +
					to_string(fliprotmat[2]) + "\t";
			}

			//Start the actual NMR calculation based on the Monte Carlo fit data
			output.resize(fluorinatedAAs.size());
			outputneg.resize(fluorinatedAAs.size());
			flipoutput.resize(fluorinatedAAs.size());
			flipoutputneg.resize(fluorinatedAAs.size());

			for (int i = 0; i < fluorinatedAAs.size(); i++)
			{

				//Time to construct the geometry vector for the residue
				//Search through and find the fluorine
				int fluorid = -1;
				for (int j = 0; j < fluorinatedAAs[i].size(); ++j)
				{
					if (fluorinatedAAs[i][j].element == "F")
					{
						fluorid = j;
						break;
					}
				}

				if (fluorid == -1)
				{
					cout << "ERROR: Couldn't find fluorine for residue " << fluorinatedAAs[i][0].resSeq << endl;
					continue;
				}

				//Find the carbon its bound to
				auto shortestdist = 10000.0f;
				int boundcarbon = -1;
				for (int j = 0; j < fluorinatedAAs[i].size(); ++j)
				{
					if (fluorinatedAAs[i][j].element == "C")
					{
						auto currdist = sqrtf(((fluorinatedAAs[i][j].x - fluorinatedAAs[i][fluorid].x) * (fluorinatedAAs[i][j].x - fluorinatedAAs[i][fluorid].x)) +
							((fluorinatedAAs[i][j].y - fluorinatedAAs[i][fluorid].y) * (fluorinatedAAs[i][j].y - fluorinatedAAs[i][fluorid].y)) +
							((fluorinatedAAs[i][j].z - fluorinatedAAs[i][fluorid].z) * (fluorinatedAAs[i][j].z - fluorinatedAAs[i][fluorid].z)));
						if (currdist < shortestdist)
						{
							shortestdist = currdist;
							boundcarbon = j;
						}
					}
				}

				if (boundcarbon == -1)
				{
					cout << "ERROR: Couldn't find a carbon to bind to fluorine for residue " << fluorinatedAAs[i][0].resSeq << endl;
					continue;
				}

				//Calculate the geometry vector
				vector<float> geomvect = {
					(fluorinatedAAs[i][fluorid].x - fluorinatedAAs[i][boundcarbon].x),
					(fluorinatedAAs[i][fluorid].y - fluorinatedAAs[i][boundcarbon].y),
					(fluorinatedAAs[i][fluorid].z - fluorinatedAAs[i][boundcarbon].z)
				};

				vector<float> flipgeomvect = {
					(flipfluorinatedAAs[i][fluorid].x - flipfluorinatedAAs[i][boundcarbon].x),
					(flipfluorinatedAAs[i][fluorid].y - flipfluorinatedAAs[i][boundcarbon].y),
					(flipfluorinatedAAs[i][fluorid].z - flipfluorinatedAAs[i][boundcarbon].z)
				};

				//Calculate Euler angles
				auto angles = generateGeometryRotationAnglesToX(geomvect);
				angles[0] *= (360.0f / (2.0f * (float)M_PI));
				angles[1] *= (360.0f / (2.0f * (float)M_PI));
				angles[2] *= (360.0f / (2.0f * (float)M_PI));
				reportlines[i] += to_string(geomvect[0]) + "\t" + to_string(geomvect[1]) + "\t" + to_string(geomvect[2]) + "\t" + to_string(angles[0]) + "\t" + to_string(angles[1]) + "\t" + to_string(angles[2]) + "\t";

				auto flipangles = generateGeometryRotationAnglesToX(flipgeomvect);
				flipangles[0] *= (360.0f / (2.0f * (float)M_PI));
				flipangles[1] *= (360.0f / (2.0f * (float)M_PI));
				flipangles[2] *= (360.0f / (2.0f * (float)M_PI));
				reportlines[i] += to_string(flipgeomvect[0]) + "\t" + to_string(flipgeomvect[1]) + "\t" + to_string(flipgeomvect[2]) + "\t" + to_string(flipangles[0]) + "\t" + to_string(flipangles[1]) + "\t" + to_string(flipangles[2]) + "\t";

				//Get the NMR shift from the Monte Carlo fit equation
				auto field = fluorines[i].getTotalField() * 10000.0f;
				auto dielectric = calculateAverageDielectric(10000, 1.0f, plaingpuatoms, plaingpuatoms[fluorinesindicies[i]], variance, inDielectric, outDielectric);
				//For testing purposes, since this is acting funny
				auto nmr = pheNMR(angles[0], angles[1], angles[2], dielectric, field);
				auto nmrneg = pheNMR(angles[0], angles[1], angles[2], dielectric, -1.0f * field);
				output[i] = nmr;
				outputneg[i] = nmrneg;
				reportlines[i] += to_string(field) + "\t" + to_string(dielectric) + "\t" + to_string(nmr) + "\t" + to_string(nmrneg) + "\t";


				//Get the NMR shift from the Monte Carlo fit equation
				auto flipfield = fluorines[i].getTotalField() * 10000.0f;
				auto flipdielectric = calculateAverageDielectric(10000, 1.0f, plaingpuatoms, plaingpuatoms[fluorinesindicies[i]], variance, inDielectric, outDielectric);
				//For testing purposes, since this is acting funny
				auto flipnmr = pheNMR(flipangles[0], flipangles[1], flipangles[2], flipdielectric, flipfield);
				auto flipnmrneg = pheNMR(flipangles[0], flipangles[1], flipangles[2], flipdielectric, -1.0f * flipfield);
				flipoutput[i] = flipnmr;
				flipoutputneg[i] = flipnmrneg;
				reportlines[i] += to_string(flipfield) + "\t" + to_string(flipdielectric) + "\t" + to_string(flipnmr) + "\t" + to_string(flipnmrneg) + "\t";

				report << reportlines[i] << endl;
			}
			// output the time took
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

int electricFieldCalculationGradientOpt(string pdbPath, const int res, const float inDielectric, const float outDielectric, const float variance, const float powvar, const float multvar, vector<float> & output, vector<float> & outputneg, vector<float> & flipoutput, vector<float> & flipoutputneg, vector<float> params)
{
	string tagline = to_string(inDielectric) + "\t" + to_string(outDielectric) + "\t" + to_string(variance) + "\t";

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
		unsigned long long memdielectricmatrix;
		memdielectricmatrix = (actres * nAtoms * sizeof(float));
		size_t freemem;
		freemem = (cudaFreeMem * 0.10f) - (nAtoms * sizeof(GPUChargeAtom)) - (nFluorines * sizeof(GPUEFP)) - memdielectricmatrix - (2 * nAtoms * sizeof(float));
		if (freemem < 0)
		{
			cout << "Error: Not enough memory for this calculation!" << endl;
			return 1;
		}
		int slicesperiter;
		slicesperiter = floor(freemem / memdielectricmatrix);
		int iterReq;
		iterReq = round(nAtoms / slicesperiter + 0.5f);
		int resopsperiter;
		resopsperiter = slicesperiter * actres;

		//Start doing the actual analysis using the Effective Length method for the non-uniform dielectric
		float* integralMatrix;
		integralMatrix = new float[nAtoms * nFluorines];
		float alpha;
		alpha = (((2.99792458f * 2.99792458f) * 1.602176487f) / (5.142206f)) * 0.1f; //Conversion factor to make calculation spit out voltage in Gaussian 09 atomic units. Analogous to Coulomb's Constant.
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
		electricFieldComponentGradOptCuda(gpuefparray, integralMatrix, gpuatomsarray, alpha, powvar, multvar, nFluorines, nAtoms, deviceProp);
		if (cudaResult != cudaSuccess)
		{
			cout << "Failed to run electric field component kernel." << endl;
			goto kernelFailed;
		}

		{
			//Get all the geometries of the fluorinated amino acids
			vector<vector<Atom>> fluorinatedAAs;
			vector<vector<Atom>> flipfluorinatedAAs;
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
				flipfluorinatedAAs.push_back(temp);
			}

			for (int i = 0; i < fluorinatedAAs.size(); i++)
			{
				vector<float> fieldVect{ fluorines[i].fieldx, fluorines[i].fieldy, fluorines[i].fieldz };
				auto rotmat = rotateResidueToXField(fieldVect, fluorinatedAAs[i]);
				rotmat[0] *= (360.0f / (2.0f * (float)M_PI));
				rotmat[1] *= (360.0f / (2.0f * (float)M_PI));
				rotmat[2] *= (360.0f / (2.0f * (float)M_PI));

				vector<float> flipfieldVect{ -1.0f * fluorines[i].fieldx, -1.0f * fluorines[i].fieldy, -1.0f * fluorines[i].fieldz };
				auto flipfieldstr = sqrtf((flipfieldVect[0] * flipfieldVect[0]) + (flipfieldVect[1] * flipfieldVect[1]) + (flipfieldVect[2] * flipfieldVect[2]));
				auto fliprotmat = rotateResidueToXField(flipfieldVect, flipfluorinatedAAs[i]);
				fliprotmat[0] *= (360.0f / (2.0f * (float)M_PI));
				fliprotmat[1] *= (360.0f / (2.0f * (float)M_PI));
				fliprotmat[2] *= (360.0f / (2.0f * (float)M_PI));
			}

			//Start the actual NMR calculation based on the Monte Carlo fit data
			output.resize(fluorinatedAAs.size());
			outputneg.resize(fluorinatedAAs.size());
			flipoutput.resize(fluorinatedAAs.size());
			flipoutputneg.resize(fluorinatedAAs.size());

			for (int i = 0; i < fluorinatedAAs.size(); i++)
			{

				//Time to construct the geometry vector for the residue
				//Search through and find the fluorine
				int fluorid = -1;
				for (int j = 0; j < fluorinatedAAs[i].size(); ++j)
				{
					if (fluorinatedAAs[i][j].element == "F")
					{
						fluorid = j;
						break;
					}
				}

				if (fluorid == -1)
				{
					cout << "ERROR: Couldn't find fluorine for residue " << fluorinatedAAs[i][0].resSeq << endl;
					continue;
				}

				//Find the carbon its bound to
				auto shortestdist = 10000.0f;
				int boundcarbon = -1;
				for (int j = 0; j < fluorinatedAAs[i].size(); ++j)
				{
					if (fluorinatedAAs[i][j].element == "C")
					{
						auto currdist = sqrtf(((fluorinatedAAs[i][j].x - fluorinatedAAs[i][fluorid].x) * (fluorinatedAAs[i][j].x - fluorinatedAAs[i][fluorid].x)) +
							((fluorinatedAAs[i][j].y - fluorinatedAAs[i][fluorid].y) * (fluorinatedAAs[i][j].y - fluorinatedAAs[i][fluorid].y)) +
							((fluorinatedAAs[i][j].z - fluorinatedAAs[i][fluorid].z) * (fluorinatedAAs[i][j].z - fluorinatedAAs[i][fluorid].z)));
						if (currdist < shortestdist)
						{
							shortestdist = currdist;
							boundcarbon = j;
						}
					}
				}

				if (boundcarbon == -1)
				{
					cout << "ERROR: Couldn't find a carbon to bind to fluorine for residue " << fluorinatedAAs[i][0].resSeq << endl;
					cout << "\t" << fluorinatedAAs[i].size() << endl;
					for (int n = 0; n < fluorinatedAAs[i].size(); ++n)
					{
						cout << "\t" << fluorinatedAAs[i][n].name << " <" << fluorinatedAAs[i][n].x << "," << fluorinatedAAs[i][n].y << "," << fluorinatedAAs[i][n].z << ">" << endl;
					}
					continue;
				}

				//Calculate the geometry vector
				vector<float> geomvect = {
					(fluorinatedAAs[i][fluorid].x - fluorinatedAAs[i][boundcarbon].x),
					(fluorinatedAAs[i][fluorid].y - fluorinatedAAs[i][boundcarbon].y),
					(fluorinatedAAs[i][fluorid].z - fluorinatedAAs[i][boundcarbon].z)
				};

				vector<float> flipgeomvect = {
					(flipfluorinatedAAs[i][fluorid].x - flipfluorinatedAAs[i][boundcarbon].x),
					(flipfluorinatedAAs[i][fluorid].y - flipfluorinatedAAs[i][boundcarbon].y),
					(flipfluorinatedAAs[i][fluorid].z - flipfluorinatedAAs[i][boundcarbon].z)
				};

				//Calculate Euler angles
				auto angles = generateGeometryRotationAnglesToX(geomvect);
				angles[0] *= (360.0f / (2.0f * (float)M_PI));
				angles[1] *= (360.0f / (2.0f * (float)M_PI));
				angles[2] *= (360.0f / (2.0f * (float)M_PI));

				auto flipangles = generateGeometryRotationAnglesToX(flipgeomvect);
				flipangles[0] *= (360.0f / (2.0f * (float)M_PI));
				flipangles[1] *= (360.0f / (2.0f * (float)M_PI));
				flipangles[2] *= (360.0f / (2.0f * (float)M_PI));

				//Get the NMR shift from the Monte Carlo fit equation
				auto field = fluorines[i].getTotalField() * 10000.0f;
				field = (params[10] * pow(field, params[9])) + params[8];

				auto dielectric = calculateAverageDielectric(10000, params[16], plaingpuatoms, plaingpuatoms[fluorinesindicies[i]], params[5], params[3], params[4]);
				dielectric = (params[13] * pow(dielectric, params[12])) + params[11];

				//For testing purposes, since this is acting funny
				auto nmr = pheNMR(angles[0], angles[1], angles[2], dielectric, field);
				nmr = (params[15] * nmr) + params[14];

				auto nmrneg = pheNMR(angles[0], angles[1], angles[2], dielectric, -1.0f * field);
				nmrneg = (params[15] * nmrneg) + params[14];

				output[i] = nmr;
				outputneg[i] = nmrneg;


				//Get the NMR shift from the Monte Carlo fit equation
				auto flipfield = fluorines[i].getTotalField() * 10000.0f;
				flipfield = (params[10] * pow(flipfield, params[9])) + params[8];


				auto flipdielectric = calculateAverageDielectric(10000, params[16], plaingpuatoms, plaingpuatoms[fluorinesindicies[i]], params[5], params[3], params[4]);
				flipdielectric = (params[13] * pow(flipdielectric, params[12])) + params[11];
				
				//For testing purposes, since this is acting funny
				auto flipnmr = pheNMR(flipangles[0], flipangles[1], flipangles[2], flipdielectric, flipfield);
				flipnmr = (params[15] * flipnmr) + params[14];

				auto flipnmrneg = pheNMR(flipangles[0], flipangles[1], flipangles[2], flipdielectric, -1.0f * flipfield);
				flipnmrneg = (params[15] * flipnmrneg) + params[14];

				flipoutput[i] = flipnmr;
				flipoutputneg[i] = flipnmrneg;
			}
			// output the time took
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

void gradientOptFunc(string pdbFilePath, int res, int optparam, float stepsize, float lasterror, vector<vector<float>> & parameters, vector<float> correctshifts, ofstream & logfile)
{
	//Copy actual parameter values
	//Variance, Reference, EFieldCompPow, EFieldCompMult
	vector<float> params;
	for (int i = 0; i < parameters.size(); ++i)
	{
		params.push_back(parameters[i][3]);
	}

	//Calculate negative direction error
	vector<float> nmrresults;
	vector<float> nmrnegresults;
	vector<float> flipnmrresults;
	vector<float> flipnmrnegresults;

	params[optparam] = parameters[optparam][3] - parameters[optparam][2];

	electricFieldCalculationGradientOpt(pdbFilePath, res, params[0], params[1], params[2], params[6], params[7], nmrresults, nmrnegresults, flipnmrresults, flipnmrnegresults, params);

	vector<float> errors = { 0.0f, 0.0f, 0.0f, 0.0f };

	for (int i = 0; i < nmrresults.size(); i++)
	{
		errors[0] += errorfunc(nmrresults[i], correctshifts[i]);
		errors[1] += errorfunc(nmrnegresults[i], correctshifts[i]);
		errors[2] += errorfunc(flipnmrresults[i], correctshifts[i]);
		errors[3] += errorfunc(flipnmrnegresults[i], correctshifts[i]);
	}
	errors[0] /= (float)nmrresults.size();
	errors[1] /= (float)nmrresults.size();
	errors[2] /= (float)nmrresults.size();
	errors[3] /= (float)nmrresults.size();

	float minuserror = *min_element(begin(errors), end(errors));

	//Calculate positive direction error
	params[optparam] = parameters[optparam][3] + parameters[optparam][2];

	electricFieldCalculationGradientOpt(pdbFilePath, res, params[0], params[1], params[2], params[6], params[7], nmrresults, nmrnegresults, flipnmrresults, flipnmrnegresults, params);

	errors = { 0.0f, 0.0f, 0.0f, 0.0f };

	for (int i = 0; i < nmrresults.size(); i++)
	{
		errors[0] += errorfunc(nmrresults[i], correctshifts[i]);
		errors[1] += errorfunc(nmrnegresults[i], correctshifts[i]);
		errors[2] += errorfunc(flipnmrresults[i], correctshifts[i]);
		errors[3] += errorfunc(flipnmrnegresults[i], correctshifts[i]);
	}
	errors[0] /= (float)nmrresults.size();
	errors[1] /= (float)nmrresults.size();
	errors[2] /= (float)nmrresults.size();
	errors[3] /= (float)nmrresults.size();

	float pluserror = *min_element(begin(errors), end(errors));

	//Get gradient of error
	float gradient = (pluserror - minuserror) / (parameters[optparam][2] * 2.0f);

	//For bad backtracking line search code
	params[optparam] = parameters[optparam][3] - gradient;
	electricFieldCalculationGradientOpt(pdbFilePath, res, params[0], params[1], params[2], params[6], params[7], nmrresults, nmrnegresults, flipnmrresults, flipnmrnegresults, params);

	errors = { 0.0f, 0.0f, 0.0f, 0.0f };

	for (int i = 0; i < nmrresults.size(); i++)
	{
		errors[0] += errorfunc(nmrresults[i], correctshifts[i]);
		errors[1] += errorfunc(nmrnegresults[i], correctshifts[i]);
		errors[2] += errorfunc(flipnmrresults[i], correctshifts[i]);
		errors[3] += errorfunc(flipnmrnegresults[i], correctshifts[i]);
	}
	errors[0] /= (float)nmrresults.size();
	errors[1] /= (float)nmrresults.size();
	errors[2] /= (float)nmrresults.size();
	errors[3] /= (float)nmrresults.size();

	float blsleft = *min_element(begin(errors), end(errors));
	float blsright = lasterror - ((parameters[optparam][2] / 2.0f) * gradient * gradient);

	if (blsleft > blsright)
	{
		cout << "Tweaking parameter step: " << parameters[optparam][2] << " => " << (parameters[optparam][2] * stepsize) << " ( " << blsleft << " > " << blsright << ")" << endl;
		logfile << "Tweaking parameter step: " << parameters[optparam][2] << " => " << (parameters[optparam][2] * stepsize) << " ( " << blsleft << " > " << blsright << ")" << endl;
		parameters[optparam][2] *= stepsize;
	}
	auto newval = parameters[optparam][3] - (parameters[optparam][2] * gradient);
	cout << parameters[optparam][3] << " => " << newval << endl;
	logfile << parameters[optparam][3] << " => " << newval << endl;
	parameters[optparam][3] = newval;
}
