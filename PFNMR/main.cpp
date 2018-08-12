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
#include <chrono>
#include <algorithm>
#include <map>
#include <random>

#include "kernel.cuh"
//#include "ProteinDisplay.h"
#include "Heatmap.h"
#include "CalculationMethods.h"
#include "TrainingMethods.h"
#include "PDBProcessor.h"
//#include "PFDProcessor.h"
#include "CSVReader.h"
#include "GPUTypes.h"
#include "gif.h"
#include "helper_string.h"

#include "GaussQuadrature.h"

using namespace std;

#define EFIELDTESTING
#define FULL_PARAM_LOG

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
        cout << "      -oD=OutDielectric (Optional, Default 80.0)" << endl;
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
    auto outDielectric = 80.0f;
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
        //electricPotentialCalculation(pdbFilePath, 100, nSlices, imgSize, inDielectric, outDielectric, relVariance);
        //createDielectricPFDFile("maptest_fine.pfd", pdbFilePath, "AtomColors.csv", 50, imgSize, outDielectric, inDielectric, relVariance);
        //ProteinDisplay display;
        //display.displayPFD("maptest_fine.pfd");
        vector<float> correctshifts;
        correctshifts.push_back( 2.51f);
        correctshifts.push_back( 2.74f);
        correctshifts.push_back( 1.19f);
        correctshifts.push_back(-0.69f);
        correctshifts.push_back(-7.13f);
        correctshifts.push_back(-4.77f);
        correctshifts.push_back( 1.77f);
        correctshifts.push_back(-1.35f);

        auto correctcenter = 0.0f;
        for (int i = 0; i < correctshifts.size(); i++)
        {
            correctcenter += correctshifts[i];
        }
        auto correctminmax = minmax_element(correctshifts.begin(), correctshifts.end());
        auto correctrange = correctshifts[correctminmax.second - correctshifts.begin()] - correctshifts[correctminmax.first - correctshifts.begin()];
        correctcenter /= correctshifts.size();

        vector<string> filestoprocess { 
			pdbFilePath
        };
        
        ofstream logfile("PFNMR-Results.log", ofstream::out);

        vector<float> totalerrors;
        totalerrors.resize(correctshifts.size());
        for (int fnum = 0; fnum < filestoprocess.size(); fnum++)
        {
            logfile << "Processing: " << filestoprocess[fnum] << endl;
            vector<float> nmrvalues;
            electricFieldCalculation(filestoprocess[fnum], 10, 4.0f, 75.0f, 0.73f, nmrvalues);

            auto calccenter = 0.0f;
            for (int i = 0; i < nmrvalues.size(); i++)
            {
                calccenter += nmrvalues[i];
            }

            auto calcminmax = minmax_element(nmrvalues.begin(), nmrvalues.end());
            auto calcrange = nmrvalues[calcminmax.second - nmrvalues.begin()] - nmrvalues[calcminmax.first - nmrvalues.begin()];
            calccenter /= nmrvalues.size();
            calccenter -= correctcenter;

            cout << "Correct range: [" << correctshifts[correctminmax.first - correctshifts.begin()] << ", " << correctshifts[correctminmax.second - correctshifts.begin()] << "]\t" << correctrange << endl;
            cout << "Predict range: [" << nmrvalues[calcminmax.first - nmrvalues.begin()] << ", " << nmrvalues[calcminmax.second - nmrvalues.begin()] << "]\t" << calcrange << endl;
            logfile << "Correct range: [" << correctshifts[correctminmax.first - correctshifts.begin()] << ", " << correctshifts[correctminmax.second - correctshifts.begin()] << "]\t" << correctrange << endl;
            logfile << "Predict range: [" << nmrvalues[calcminmax.first - nmrvalues.begin()] << ", " << nmrvalues[calcminmax.second - nmrvalues.begin()] << "]\t" << calcrange << endl;

            auto averageerror = 0.0f;

            cout << "Offset value: " << calccenter << endl;
            logfile << "Offset value: " << calccenter << endl;
            for (int i = 0; i < nmrvalues.size(); i++)
            {
                nmrvalues[i] -= calccenter;
                auto error = ((nmrvalues[i] - correctshifts[i]) / correctshifts[i]) * 100.0f;
                totalerrors[i] += abs(error);
                averageerror += abs(error);
                cout << "[" << i << "]: " << nmrvalues[i] << " (" << correctshifts[i] << ")\t" << error << "%" << endl;
                logfile << "[" << i << "]: " << nmrvalues[i] << " (" << correctshifts[i] << ")\t" << error << "%" << endl;
            }
            cout << endl;
            logfile << endl;
            averageerror /= (float)nmrvalues.size();
            cout << "\t\tAverage Error: " << averageerror << "%" << endl;
            cout << endl << endl;
            logfile << "\t\tAverage Error: " << averageerror << "%" << endl;
            logfile << endl << endl;
        }

        auto totalaverage = 0.0f;
        for (int i = 0; i < totalerrors.size(); i++)
        {
            totalaverage += totalerrors[i];
        }
        totalaverage /= (float)(filestoprocess.size() * correctshifts.size());
        cout << "\t\tTotal Average Error: " << totalaverage << "%" << endl;
        logfile << "\t\tTotal Average Error: " << totalaverage << "%" << endl;
        for (int i = 0; i < totalerrors.size(); i++)
        {
            totalerrors[i] /= (float)filestoprocess.size();
            cout << "[" << i << "]: " << totalerrors[i] << "%" << endl;
        }
        cin.get();
        return 0;
    }

	int gradientsteps = 1;
	int maxsteps = 20;

	if (checkCmdLineFlag(argc, (const char**)argv, "gradsteps"))
	{
		gradientsteps = getCmdLineArgumentInt(argc, (const char**)argv, "gradsteps");

		if (gradientsteps < 1)
		{
			cout << "Error: Value for gradient steps must be greater than 1." << endl;
			cout << "Exiting..." << endl;

			return 1;
		}
	}

	if (checkCmdLineFlag(argc, (const char**)argv, "maxsteps"))
	{
		maxsteps = getCmdLineArgumentInt(argc, (const char**)argv, "maxsteps");

		if (maxsteps < 1)
		{
			cout << "Error: Value for max optimzation steps must be greater than 1." << endl;
			cout << "Exiting..." << endl;

			return 1;
		}
	}


	if (checkCmdLineFlag(argc, (const char**)argv, "gradientparam"))
	{
		ofstream logfile("PFNMR-GradOptResults.log", ofstream::out);
		//Uses concepts from: https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf
		//PARAMETERS TO OPTIMIZE
		vector<vector<float>> params = {
			{ 1.0f, 30.0f, 0.1f, 23.2961f },	//[0] Inner Dielectric
			{ 70.0f, 90.0f, 0.1f, 86.9119f },	//[1] Outer Dielectric
			{ 0.10f, 2.50f, 0.01f, 0.221146f },	//[2] Variance
			{ 1.0f, 30.0f, 0.1f, 21.2f },	//[3] Inner Dielectric 2
			{ 70.0f, 90.0f, 0.1f, 89.84f },	//[4] Outer Dielectric 2
			{ 0.10f, 2.50f, 0.01f, 1.79988f },	//[5] Variance 2
			{ 0.8f, 1.2f, 0.01f, 0.893421f },	//[6] EField Component Power
			{ 0.8f, 1.2f, 0.1f, 0.982078f },		//[7] EField Component Multiplier
			{ -1.0f, 1.0f, 0.1f, 0.00250125f },	//[8] EField Offset
			{ 0.8f, 1.2f, 0.01f, 0.917875f },	//[9] EField Power
			{ 0.8f, 1.2f, 0.1f, 1.00828f },		//[10]EField Multiplier
			{ -1.0f, 1.0f, 0.1f, -0.300015f },	//[11]Dielectric Offset
			{ 0.8f, 1.2f, 0.01f, 0.949967f },	//[12]Dielectric Power
			{ 0.8f, 1.2f, 0.1f, 1.09985f },		//[13]Dielectric Multiplier
			{ -1.0f, 1.0f, 0.01f, -0.57f },	//[14]NMR Offset
			{ 0.8f, 1.2f, 0.1f, 1.0274f },		//[15]NMR Multiplier
			{ 0.5f, 2.0f, 0.1f, 0.899856f }		//[16]DielectricShell
		};
		vector<string> paramnames = { 
			"Inner Dielectric",
			"Outer Dielectric",
			"Variance",
			"Inner Dielectric 2",
			"Outer Dielectric 2",
			"Variance 2",
			"EField Component Power",
			"EField Component Multiplier",
			"EField Offset",
			"EField Power",
			"EField Multiplier",
			"Dielectric Offset",
			"Dielectric Power",
			"Dielectric Multiplier",
			"NMR Offset",
			"NMR Multiplier",
			"DielectricShell"
		};
		vector<int> paramstoopt = {};

		//Relevant parameters
		auto stepsize = 0.8f;
		auto convergencethreshold = 0.001f;
		auto res = 10;

		//Correct chemical shift values for IFABP
		map<int, float> IFABPShifts = {
			{2  ,  2.51f},
			{17 ,  2.74f},
			{47 ,  1.19f},
			{55 , -0.69f},
			{62 , -7.13f},
			{68 , -4.77f},
			{93 ,  1.77f},
			{128, -1.35f}
		};
		vector<int> rescheck = { 2, 17, 47, 55, 62, 68, 93, 128 };

		CSVReader csv("ChargeTable.csv");
		auto chargetable = csv.readCSVFile();
		PDBProcessor pdb(pdbFilePath);
		auto baseatoms = pdb.getAtomsFromPDB();
		auto gpuatoms = pdb.getGPUChargeAtomsFromAtoms(baseatoms, chargetable);
		vector<float> weights;
		vector<float> abscissa;
		getGaussQuadSetup(20, weights, abscissa);
		if (baseatoms.size() == 0)
		{
			cout << "ERROR: No atoms found!" << endl;
			cin.get();
			return 1;
		}

		vector<GPUEFP> fluorines;
		vector<float> correctshifts;
		
		for (int i = 0; i < baseatoms.size(); i++)
		{
			if (baseatoms[i].element == "F")
			{
				if (find(rescheck.begin(), rescheck.end(), baseatoms[i].resSeq) != rescheck.end())
				{

					GPUEFP newefp;
					newefp.x = baseatoms[i].x;
					newefp.y = baseatoms[i].y;
					newefp.z = baseatoms[i].z;
					newefp.chainid = (int)baseatoms[i].chainID;
					newefp.resid = baseatoms[i].resSeq;
					fluorines.push_back(newefp);
					correctshifts.push_back(IFABPShifts[newefp.resid]);
				}
				else
				{
					cout << "ERROR: Fluorine on residue " << baseatoms[i].resSeq << " doesn't correspond to a correct chemical shift test value!" << endl;
					cout << "Exitting..." << endl;
					return 1;
				}
			}
		}
		
		// find out how much we can calculate
		cudaDeviceProp deviceProp;
		cudaError_t cudaResult;
		cudaResult = cudaGetDeviceProperties(&deviceProp, 0);

		if (cudaResult != cudaSuccess)
		{
			cerr << "cudaGetDeviceProperties failed!" << endl;
			return 1;
		}
//===================================================================================================================================================
		int gradcounter = 0;
		float bestoptimizationerror = FLT_MAX;
		vector<float> bestparams;
		for (int i = 0; i < paramstoopt.size(); ++i)
		{
			bestparams.push_back(0.0f);
		}
		do
		{
			cout << "===================================================================================================================================================" << endl;
			cout << "Trial: " << (gradcounter + 1) << endl;
			logfile << "===================================================================================================================================================" << endl;
			logfile << "Trial: " << (gradcounter + 1) << endl;
			//Initialize initial parameter states randomly
			cout << "Initializing with the following values:" << endl;
			logfile << "Initializing with the following values:" << endl;
			random_device rd;
			mt19937 e2(rd());
			uniform_real_distribution<float> dist(0.0f, 1.0f);
			for (int i = 0; i < paramstoopt.size(); ++i)
			{
				params[paramstoopt[i]][3] = rounderhelper((dist(e2) * (params[paramstoopt[i]][1] - params[paramstoopt[i]][0])) + params[paramstoopt[i]][0], params[paramstoopt[i]][2]);
				cout << paramnames[paramstoopt[i]] << ": " << params[paramstoopt[i]][3] << endl;
				logfile << paramnames[paramstoopt[i]] << ": " << params[paramstoopt[i]][3] << endl;
			}
			cout << endl;
			logfile << endl;


			//Calculate base error
			vector<float> currparamlist;
			vector<float> nmrresults;
			vector<float> nmrnegresults;
			vector<float> flipnmrresults;
			vector<float> flipnmrnegresults;

			for (int i = 0; i < params.size(); ++i)
			{
				currparamlist.push_back(params[i][3]);
			}
			electricFieldCalculationGradientOpt(pdbFilePath, res, params[0][3], params[1][3], params[2][3], params[6][3], params[7][3], nmrresults, nmrnegresults, flipnmrresults, flipnmrnegresults, currparamlist);
			
			for (int i = 0; i < nmrresults.size(); ++i)
			{
				cout << fluorines[i].resid << "\t" <<  nmrresults[i] << ":" << nmrnegresults[i] << ":" << flipnmrresults[i] << ":" << flipnmrnegresults[i] << endl;
			}
			
			return 1;
			vector<float> errors = { 0.0f, 0.0f, 0.0f, 0.0f };
			float temperr = 0.0f;
			float tempnegerr = 0.0f;
			float tempfliperr = 0.0f;
			float tempnegfliperr = 0.0f;
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

			float lasterror = *min_element(begin(errors), end(errors));
			cout << "Initial error: " << lasterror << endl;
			logfile << "Initial error: " << lasterror << endl;


			float errordiff = FLT_MAX;
			vector<string> stateflags = { "R", "N", "F", "FN" };
			string state = "UNK";
			int counter = 0;

			//Run the paramterization
			clock_t startTime = clock();
			do
			{
				for (int i = 0; i < paramstoopt.size(); ++i)
				{
					cout << "Optimizing " << paramnames[paramstoopt[i]] << endl;
					logfile << "Optimizing " << paramnames[paramstoopt[i]] << endl;
					gradientOptFunc(pdbFilePath, res, paramstoopt[i], stepsize, lasterror, params, correctshifts, logfile);
				}
				//==============Utilize Optimized Parameters=====================================================================================
				//Calculate new error
				for (int i = 0; i < params.size(); ++i)
				{
					currparamlist.push_back(params[i][3]);
				}
				electricFieldCalculationGradientOpt(pdbFilePath, res, params[0][3], params[1][3], params[2][3], params[6][3], params[7][3], nmrresults, nmrnegresults, flipnmrresults, flipnmrnegresults, currparamlist);

				errors = { 0.0f, 0.0f, 0.0f, 0.0f };
				temperr = 0.0f;
				tempnegerr = 0.0f;
				tempfliperr = 0.0f;
				tempnegfliperr = 0.0f;

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

				auto minerrorptr = min_element(begin(errors), end(errors));
				float finalerror = *minerrorptr;
				int pos = distance(begin(errors), minerrorptr);
				state = stateflags[pos];
				errordiff = abs(finalerror - lasterror);

				cout << "Opt " << counter << ": " << lasterror << " => " << finalerror << " (" << errordiff << ") " << endl;
				cout << "State: " << state << endl;
				logfile << "Opt " << counter << ": " << lasterror << " => " << finalerror << " (" << errordiff << ") " << endl;
				logfile << "State: " << state << endl;
				cout << endl << endl;
				lasterror = finalerror;
				++counter;
			} while ((errordiff > convergencethreshold) && (counter < maxsteps));

			cout << endl << endl << endl << endl;
			cout << "--------------------RESULTS--------------------" << endl;
			cout << "Opt " << counter << ": " << lasterror << " (" << errordiff << ") " << endl;
			cout << "State: " << state << endl;
			logfile << "--------------------RESULTS--------------------" << endl;
			logfile << "Opt " << counter << ": " << lasterror << " (" << errordiff << ") " << endl;
			logfile << "State: " << state << endl;
			for (int i = 0; i < paramstoopt.size(); ++i)
			{
				cout << paramnames[paramstoopt[i]] << ": " << params[paramstoopt[i]][3] << endl;
				logfile << paramnames[paramstoopt[i]] << ": " << params[paramstoopt[i]][3] << endl;
			}


			if (lasterror < bestoptimizationerror)
			{
				bestoptimizationerror = lasterror;
				cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW BEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
				logfile << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW BEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
				cout << "New shifts:" << endl;
				logfile << "New shifts:" << endl;
				if (state == stateflags[0])
				{
					for (int i = 0; i < nmrresults.size(); ++i)
					{
						cout << nmrresults[i] << " (" << correctshifts[i] << ")" << endl;
						logfile << nmrresults[i] << " (" << correctshifts[i] << ")" << endl;
					}
				}
				if (state == stateflags[1])
				{
					for (int i = 0; i < nmrnegresults.size(); ++i)
					{
						cout << nmrnegresults[i] << " (" << correctshifts[i] << ")" << endl;
						logfile << nmrnegresults[i] << " (" << correctshifts[i] << ")" << endl;
					}

				}
				if (state == stateflags[2])
				{
					for (int i = 0; i < flipnmrresults.size(); ++i)
					{
						cout << flipnmrresults[i] << " (" << correctshifts[i] << ")" << endl;
						logfile << flipnmrresults[i] << " (" << correctshifts[i] << ")" << endl;
					}

				}
				if (state == stateflags[3])
				{
					for (int i = 0; i < flipnmrnegresults.size(); ++i)
					{
						cout << flipnmrnegresults[i] << " (" << correctshifts[i] << ")" << endl;
						logfile << flipnmrnegresults[i] << " (" << correctshifts[i] << ")" << endl;
					}

				}

				cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW BEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
				logfile << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW BEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
			}

			++gradcounter;
		}while (gradcounter < gradientsteps);
		cout << "DONE WITH PARAMTERIZATION!" << endl;

		cin.get();
		return 0;
	}

    if (checkCmdLineFlag(argc, (const char**)argv, "parameterize"))
    {
        vector<float> correctshifts;
        correctshifts.push_back( 2.51f);
        correctshifts.push_back( 2.74f);
        correctshifts.push_back( 1.19f);
        correctshifts.push_back(-0.69f);
        correctshifts.push_back(-7.13f);
        correctshifts.push_back(-4.77f);
        correctshifts.push_back( 1.77f);
        correctshifts.push_back(-1.35f);

        CSVReader csv("ChargeTable.csv");
        auto chargetable = csv.readCSVFile();
        PDBProcessor pdb(pdbFilePath);
        auto baseatoms = pdb.getAtomsFromPDB();
        auto gpuatoms = pdb.getGPUChargeAtomsFromAtoms(baseatoms, chargetable);
        vector<float> weights;
        vector<float> abscissa;
        getGaussQuadSetup(20, weights, abscissa);

        if (baseatoms.size() == 0)
        {
            cout << "ERROR: No atoms found!" << endl;
            cin.get();
            return 1;
        }

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

        // find out how much we can calculate
        cudaDeviceProp deviceProp;
        cudaError_t cudaResult;
        cudaResult = cudaGetDeviceProperties(&deviceProp, 0);

        if (cudaResult != cudaSuccess)
        {
            cerr << "cudaGetDeviceProperties failed!" << endl;
            return 1;
        }

        //TESTING PARAMETERS
        auto maxvariance = 2.5f;
        auto minvariance = 0.1f;
        auto stepsvariance = 120.0f;
        auto maxref = 50.0f;
        auto minref = 1.0f;
        auto stepsref = 24.5f;

        //Calculated parameters
        auto varstep = (maxvariance - minvariance) / stepsvariance;
        auto refstep = (maxref - minref) / stepsref;
		int totalnumsims = (stepsvariance+1.0f) * (stepsref+1.0f);
#ifdef FULL_PARAM_LOG
		cout << "This will generate " << (stepsref * stepsvariance * 8) << " lines.  " << endl;
		ofstream paramlog("FullParamsReadout.txt", ofstream::out);
		paramlog << "inDielectric\toutDielectric\tVariance\tresid\tfieldX\tfieldY\tfieldZ\tfieldTotal\tfieldRotX\tfieldRotY\tfieldRotZ\tflipfieldX\tflipfieldY\tflipfieldZ\tflipfieldTotal\tflipfieldRotX\tflipfieldRotY\tflipfieldRotZ\tgeomX\tgeomY\tgeomZ\tgeomRotX\tgeomRotY\tgeomRotZ\tflipgeomX\tflipgeomY\tflipgeomZ\tflipgeomRotX\tflipgeomRotY\tflipgeomRotZ\tfitField\tfitDielectric\tfitNMR\tfitNMRneg\tfitflipField\tfitflipDielectric\tfitflipNMR\tfitflipNMRneg" << endl;
#endif

        ofstream paramfile("Parameterization.csv", ofstream::out);
		paramfile << "Parameterization results:" << endl;
		paramfile << "Row: Variance.  Column: Internal Reference" << endl;

        float lowerror = FLT_MAX;
        float lowerrper = FLT_MAX;
        float bestvar = 0.0f;
        float bestref = 0.0f;
		string sign = "UNK";
        //Print out the reference dielectric header
		paramfile << "  ,";
        for (float ref = minref; ref <= maxref; ref += refstep)
        {
			paramfile << ref << ",";
        }
		paramfile << endl;
        //Run the paramterization
		clock_t startTime = clock();
		int counter = 0;
        for (float var = minvariance; var <= maxvariance; var+=varstep)
        {
			paramfile << var << ",";
            for (float ref = minref; ref <= maxref; ref += refstep)
            {
                vector<float> nmrresults;
				vector<float> nmrnegresults;
				vector<float> flipnmrresults;
				vector<float> flipnmrnegresults;
#ifdef FULL_PARAM_LOG
				electricFieldCalculationSilentReporter(pdbFilePath, 10, ref, outDielectric, var, nmrresults, nmrnegresults, flipnmrresults, flipnmrnegresults, paramlog);
#else
				electricFieldCalculationSilent(pdbFilePath, 10, ref, outDielectric, var, nmrresults);
#endif
                float error = 0.0f;
                float errper = 0.0f;
                for (int i = 0; i < nmrresults.size(); i++)
                {
                    error += abs(nmrresults[i] - correctshifts[i]);
                    errper += abs((nmrresults[i] - correctshifts[i]) / correctshifts[i]) * 100.0f;
                }
                error /= (float)nmrresults.size();
                errper /= (float)nmrresults.size();
                if (error < lowerror)
                {
                    lowerror = error;
                    lowerrper = errper;
                    bestvar = var;
                    bestref = ref;
					sign = "POS";
                }

				float errorneg = 0.0f;
                float errperneg = 0.0f;
				for (int i = 0; i < nmrnegresults.size(); i++)
				{
					errorneg += abs(nmrnegresults[i] - correctshifts[i]);
                    errperneg += abs((nmrnegresults[i] - correctshifts[i]) / correctshifts[i]) * 100.0f;
				}
				errorneg /= (float)nmrnegresults.size();
                errperneg /= (float)nmrnegresults.size();
				if (errorneg < lowerror)
				{
					lowerror = errorneg;
                    lowerrper = errperneg;
					bestvar = var;
					bestref = ref;
					sign = "NEG";
				}


				float fliperror = 0.0f;
                float fliperrper = 0.0f;
				for (int i = 0; i < flipnmrresults.size(); i++)
				{
					fliperror += abs(flipnmrresults[i] - correctshifts[i]);
                    fliperrper += abs((flipnmrresults[i] - correctshifts[i]) / correctshifts[i]) * 100.0f;
				}
				fliperror /= (float)flipnmrresults.size();
				fliperrper /= (float)flipnmrresults.size();
				if (fliperror < lowerror)
				{
                    lowerrper = fliperrper;
					lowerror = fliperror;
					bestvar = var;
					bestref = ref;
					sign = "FLIP-POS";
				}

				float fliperrorneg = 0.0f;
				float fliperrperneg = 0.0f;
				for (int i = 0; i < flipnmrnegresults.size(); i++)
				{
					fliperrorneg += abs(flipnmrnegresults[i] - correctshifts[i]);
                    fliperrperneg += abs((flipnmrnegresults[i] - correctshifts[i]) / correctshifts[i]) * 100.0f;
				}
				fliperrorneg /= (float)flipnmrnegresults.size();
				fliperrperneg /= (float)flipnmrnegresults.size();
				if (fliperrorneg < lowerror)
				{
                    lowerrper = fliperrperneg;
					lowerror = fliperrorneg;
					bestvar = var;
					bestref = ref;
					sign = "FLIP-NEG";
				}


                cout << endl;
                cout << "Error: " << error << " | " << fliperror << " ppm\tRef: " << ref << "\tVar: " << var << endl;
				cout << "ErrorNeg: " << errorneg << " | " << fliperrorneg << " ppm\tRef: " << ref << "\tVar: " << var << endl;
                cout << "Best error: " << lowerror << " ppm (" << lowerrper << ")\tRef: " << bestref << "\tVar: " << bestvar << "\tFlag: " << sign << endl;
				++counter;
				auto elapsedtime = ((clock() - startTime) / ((double)CLOCKS_PER_SEC));
				auto remaining = ((double)totalnumsims - (double)counter) * (elapsedtime / (double)counter);
				cout << "Time remaining: (" << counter << "/" << totalnumsims << "): " << remaining << endl;

				paramfile << error << ",";
            }
			paramfile << endl;
        }
        cout << "DONE WITH PARAMTERIZATION!" << endl;
        cout << "Best results- Ref:" << bestref << "\tVar:" << bestvar << endl;
        cin.get();
        return 0;
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
        size_t nGpuGridPointBase;
        nGpuGridPointBase = floor((cudaFreeMem * 0.45f - (nAtoms * sizeof(GPUAtom))) / ((nAtoms * sizeof(float)) + sizeof(GridPoint)));
        int itersReq;
        itersReq = round(imgSizeSq / nGpuGridPointBase + 0.5f); // pull some computer math bs to make this work
        GridPoint* gridPoints;
        gridPoints = new GridPoint[imgSizeSq];

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
