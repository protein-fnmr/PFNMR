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

// C++ code for implementing KAKSI method for secondary structure determination
// Method taken from: "Protein secondary structure assignment revisited: a detailed analysis of different assignment methods", 
//                           Juliette Martin et. al, BMC Structural Biology, doi:10.1186/1472-6807-5-17

#define _USE_MATH_DEFINES

#include <math.h>

#include "KAKSI.h"

//Useful constants for the calculation, taken from the paper
//Alpha Helix parameters
#define H_II2_CORE_MEAN 5.49
#define H_II2_CORE_SD   0.20
#define H_II2_TERM_MEAN 5.54
#define H_II2_TERM_SD   0.25
#define H_II3_CORE_MEAN 5.30
#define H_II3_CORE_SD   0.64
#define H_II3_TERM_MEAN 5.36
#define H_II3_TERM_SD   0.39
#define H_II4_MEAN      6.33
#define H_II4_SD        0.71
#define H_II5_MEAN      8.72
#define H_II5_SD        0.63
#define H_PHI_MEAN      -64.0
#define H_PHI_SD        7.0
#define H_PSI_MEAN      -41.0
#define H_PSI_SD        7.0
#define H_MIN_LENGTH    5
//Parallel Sheet parameters
#define P_IJ_MEAN       4.83
#define P_IJ_SD         0.29
#define P_I1J1_MEAN     4.84
#define P_I1J1_SD       0.24
#define P_I2J1_MEAN     6.07
#define P_I2J1_SD       0.35
#define P_II2_MEAN      6.70
#define P_II2_SD        0.32
//Anti-parallel Sheet parameters
#define A_IJ_MEAN       4.77
#define A_IJ_SD         0.42
#define A_I1J1_MEAN     4.88
#define A_I1J1_SD       0.43
#define A_I2J1_MEAN     6.00
#define A_I2J1_SD       0.47
#define A_II2_MEAN      6.70
#define A_II2_SD        0.32
//Calculation parameters
#define EPSILON_H       1.96
#define ETA_H           2.25
#define EPSILON_B       2.58
#define SIGMA_B         5.00
//Sliding Window parameters
#define W1              6
#define W2              4
#define W3              3

float radtodeg(float rad)
{
    return (rad * (180.0f / M_PI));
}

float dotProd(vector<float> & a, vector<float> & b)
{
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

vector<float> crossProd(vector<float> & a, vector<float> & b)
{
    vector<float> result;
    result.push_back((a[1] * b[2]) - (a[2] * b[1]));
    result.push_back((a[2] * b[0]) - (a[0] * b[2]));
    result.push_back((a[0] * b[1]) - (a[1] * b[0]));
    return result;
}

vector<float> distance(Atom & a, Atom & b)
{
    vector<float> result;
    result.push_back(b.x - a.x);
    result.push_back(b.y - a.y);
    result.push_back(b.z - a.z);
    return result;
}

float magnitude(vector<float> & a)
{
    return sqrtf((a[0] * a[0]) + (a[1] * a[1]) + (a[2] * a[2]));
}

vector<float> normalize(vector<float> & a)
{
    vector<float> result;
    float mag = magnitude(a);
    result.push_back(a[0] / mag);
    result.push_back(a[1] / mag);
    result.push_back(a[2] / mag);
    return result;
}

float dihedral(Atom & a1, Atom & a2, Atom & a3, Atom & a4)
{
    //Relevant bond vectors
    vector<float> b1, b2, b3;
    b1 = distance(a1, a2);
    b2 = distance(a2, a3);
    b3 = distance(a3, a4);

    //Calculate params needed
    auto n1 = normalize(crossProd(b1, b2));
    auto n2 = normalize(crossProd(b2, b3));

    auto x = dotProd(n1, n2);
    auto y = dotProd(crossProd(n1, normalize(b2)), n2);

    return radtodeg(atan2f(y, x));
}

//Returns a character array for each atom indicating its secondary structure ('a' = alpha helix, 'b' = beta sheet, 'l' = loop)
bool determineSecondaryStructureCPU(vector<Atom> & atoms, vector<vector<Atom>> & out_helicies, vector<vector<Atom>> & out_sheets)
{
    //Get a list of the relevant carbons
    
    vector<vector<Atom>> backbone; //[N, Ca, C(O)]
    
    for (int i = 0; i < atoms.size(); i++)
    {
        auto resnum = atoms[i].resSeq - 1;
        int cond = backbone.size() - 1;
        if (cond < resnum) //Add a dummy atom array for each residue
        {
            Atom temp;
            vector<Atom> temp2;
            backbone.push_back(temp2);
            backbone[resnum].push_back(temp);
            backbone[resnum].push_back(temp);
            backbone[resnum].push_back(temp);
        }
        if (atoms[i].name == "N")
        {
            backbone[resnum][0] = atoms[i];
        }
        else if (atoms[i].name == "CA")
        {
            backbone[resnum][1] = atoms[i];
        }
        else if (atoms[i].name == "C")
        {
            backbone[resnum][2] = atoms[i];
        }
    }

    //Setup the secondary structure flag array as "Everything is a loop"
    auto rescount = backbone.size();
    char* ss = new char[rescount];
    for (int i = 0; i < rescount; i++)
    {
        ss[i] = 'L';
    }

    //Step 1: Determine initial helices use C1 criterion
    
    for (int num = 0; num < rescount - W1; num++) //Cycle through every atom
    {
        bool flag = true;
        for (int jump = 2; (jump < W1) && flag; jump++) //Cycle through each spacing type (2 is min)
        {
            for (int start = 0; (start < (W1 - jump)) && flag; start++) //Cycle through the available atoms, given the spacing
            {
                int target = start + jump;  //The target atom position in the window
                auto dist = magnitude(distance(backbone[num + start][1], backbone[num + target][1])); //Distance between the start and target atoms in the window
                //Establish which criteria to use
                float mean = 0.0;
                float sd = 0.0;
                switch (jump)
                {
                case 2:
                    if(start == 0 || target == (W1 - 1))
                    {
                        mean = H_II2_TERM_MEAN;
                        sd = H_II2_TERM_SD;
                    }
                    else
                    {
                        mean = H_II2_CORE_MEAN;
                        sd = H_II2_CORE_SD;
                    }
                    break;
                case 3:
                    if (start == 0 || target == (W1 - 1))
                    {
                        mean = H_II3_TERM_MEAN;
                        sd = H_II3_TERM_SD;
                    }
                    else
                    {
                        mean = H_II3_CORE_MEAN;
                        sd = H_II3_CORE_SD;
                    }
                    break;
                case 4:
                    mean = H_II4_MEAN;
                    sd = H_II4_SD;
                    break;
                case 5:
                    mean = H_II5_MEAN;
                    sd = H_II5_SD;
                    break;
                default:
                    printf("Error: KAKSI found a helix distance comparison that has no parameters.  Exitting...");
                    return false;
                }
                //Check if the distance discovered matches the criterion, and flag if it doesn't
                flag = flag && ((dist < (mean + (sd * EPSILON_H))) && (dist > (mean - (sd * EPSILON_H))));
            }
        }
        //If we passed all the tests from above, flag the residues as an alpha helix
        if (flag)
        {
            for (int i = 0; i < W1; i++)
            {
                ss[num + i] = 'H';
            }
        }
    }

    //Step 2: Determine more initial helix candidates using phi-psi angles
    //Get the phi-psi for all the non-termini residues
    vector<vector<float>> phipsi;
    for (int i = 1; i < rescount - 1; i++)
    {
        //TODO: This might break, double check it is producing resonable values.
        phipsi.push_back({ dihedral(backbone[i - 1][2], backbone[i][0], backbone[i][1], backbone[i][2]), dihedral(backbone[i][0], backbone[i][1], backbone[i][2], backbone[i + 1][0]) });
    }
    
    for (int num = 0; num < phipsi.size() - W2; num++)
    {
        bool flag = true;
        bool ramaflag = false;
        //Cycle through the window, and check if the phi-psi angles are good.
        //Also check if 1 window value set is near the average Ramachandran values
        for (int i = 0; (i < W2) && flag; i++)
        {
            //Check if the value is in the ballpark
            flag = flag && ((phipsi[num + i][0] < 0.0f) && ((phipsi[num + i][1] < 60.0f) && (phipsi[num + i][1] > -90.0f)));
            //Check if the value is optimal
            ramaflag = ramaflag || ((phipsi[num + i][0] < (H_PHI_MEAN + H_PHI_SD)) && (phipsi[num + i][0] > (H_PHI_MEAN - H_PHI_SD)) && (phipsi[num + i][1] < (H_PSI_MEAN + H_PSI_SD)) && (phipsi[num + i][1] > (H_PSI_MEAN - H_PSI_SD)));
        }

        //If both criteria are good, then label the window residues as an alpha helix
        if (flag && ramaflag)
        {
            for (int i = 0; i < W2; i++)
            {
                ss[num + i] = 'H';
            }
        }
    }

    //Step 3: Remove any helix definitions that are kinks
    //For the first pass, remove all internal residues that are kinked based on phi-psi angles
    for (int num = 1; num < rescount - 3; num++)
    {
        if (&ss[num] == "HHH") //Make sure the 3-member window is helical
        {
            if (ss[num - 1] == 'H' && ss[num + 3] == 'H') //Make sure we are in the middle of a helix
            {
                //Test the criteria, and set the center window position to a loop if it shows to be kninked
                if (sqrtf(((phipsi[num][0] - phipsi[num + 1][0]) * (phipsi[num][0] - phipsi[num + 1][0])) + ((phipsi[num + 1][0] - phipsi[num + 2][0]) * (phipsi[num + 1][0] - phipsi[num + 2][0]))) +
                    sqrtf(((phipsi[num][1] - phipsi[num + 1][1]) * (phipsi[num][1] - phipsi[num + 1][1])) + ((phipsi[num + 1][1] - phipsi[num + 2][1]) * (phipsi[num + 1][1] - phipsi[num + 2][1]))) > 112.5)
                {
                    ss[num + 1] = 'L';
                }
            }
        }
    }

    //Get rid of short helix reasons (Less than a certain number of residues)
    for (int num = 0; num < rescount; num++)
    {
        int count = 0;
        if (ss[num] == 'H') //Keep going till we hit a helix, and start counting
        {
            count++;
        }
        if (ss[num] == 'L' && count > 0) //We found the end of a helix
        {
            if (count < H_MIN_LENGTH) //We found a helix that should be removed
            {
                for (int i = 0; i < count; i++)
                {
                    ss[num - i] = 'L';
                }
            }
            count = 0; //Reset and start looking again
        }
    }
    
    //For the second pass, check if the helix is better modeled by 2+ helicies by doing axis fitting
    
    //TODO: Implement the rest of the secondary structure analysis (I can feel my soul dying slowly...)

    //Harvest the secondary structure information
    bool state = false;
    for (int i = 0; i < rescount; i++)
    {
        if (ss[i] == 'H' && !state)
        {
            vector<Atom> temp;
            out_helicies.push_back(temp);
            out_helicies[out_helicies.size() - 1].push_back(backbone[i][1]);
            state = true;
        }
        if (ss[i] != 'H' && state)
        {
            out_helicies[out_helicies.size() - 1].push_back(backbone[i - 1][1]);
            state = false;
        }
    }
    if (ss[rescount] == 'H')
    {
        out_helicies[out_helicies.size() - 1].push_back(backbone[rescount - 1][1]);
    }

    for (int i = 0; i < rescount; i++)
    {
        if (ss[i] == 'S' && !state)
        {
            vector<Atom> temp;
            out_helicies.push_back(temp);
            out_helicies[out_helicies.size() - 1].push_back(backbone[i][1]);
            state = true;
        }
        if (ss[i] != 'S' && state)
        {
            out_helicies[out_helicies.size() - 1].push_back(backbone[i - 1][1]);
            state = false;
        }
    }
    if (ss[rescount] == 'S')
    {
        out_helicies[out_helicies.size() - 1].push_back(backbone[rescount - 1][1]);
    }


    return true;
}