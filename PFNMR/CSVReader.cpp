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

// C++ code used to read the CSV charge file

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <array>

#include "CSVReader.h"

using namespace std;

vector<array<string,2>> chargedata;

CSVReader::CSVReader() 
{
    cout << "CSVReader loaded." << endl;
}

//Splitting code found on http://stackoverflow.com/questions/30797769/splitting-a-string-but-keeping-empty-tokens-c
void split(const string& str, vector<string>& tokens, const string& delimiters)
{
    // Start at the beginning
    string::size_type lastPos = 0;
    // Find position of the first delimiter
    string::size_type pos = str.find_first_of(delimiters, lastPos);

    // While we still have string to read
    while (string::npos != pos && string::npos != lastPos)
    {
        // Found a token, add it to the vector
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Look at the next token instead of skipping delimiters
        lastPos = pos + 1;
        // Find the position of the next delimiter
        pos = str.find_first_of(delimiters, lastPos);
    }

    // Push the last token
    tokens.push_back(str.substr(lastPos, pos - lastPos));
}

vector<string> split(const string &s, const string& delimiters) {
    vector<string> elems;
    split(s, elems, delimiters);
    return elems;
}

// TODO: It might be useful, once working on the command line GUI, to keep the charge csv file open so the use can be shown the header line and define the relevant column positions, then close the file out afterwards.  

int CSVReader::getHeaderInformation(string chargefilepath)
{
    if (chargefilepath.find_last_of(".") == string::npos) //Make sure a file was actually passed
    {
        cout << "Error: Charge CSV \"file\" passed doesn't appear to have an extension." << endl;
        return 1;
    }
    if (!chargefilepath.substr(chargefilepath.find_last_of(".") + 1).compare(".csv")) //Make sure it's a .csv file
    {
        cout << "Error: Charge CSV file is not a .csv file." << endl;
        return 1;
    }
    //Start reading through the file and adding amino acids to the list
    ifstream csvfile(chargefilepath);
    if (csvfile.is_open()) //Can we open the file?
    {
        string line;
        getline(csvfile, line);
        auto linecontents = split(line,",");
        cout << line.size() << endl;
        for (int i = 0; i < linecontents.size(); ++i)
        {
            cout << "[" << i << "]: " << linecontents[i] << endl;
        }
    }
    csvfile.close();
}

int CSVReader::readChargeCSV(string chargefilepath, int residuecolumn, int atomcolumn, int chargecolumn)
{
    if (chargefilepath.find_last_of(".") == string::npos) //Make sure a file was actually passed
    {
        cout << "Error: Charge CSV \"file\" passed doesn't appear to have an extension." << endl;
        return 1;
    }
    if (!chargefilepath.substr(chargefilepath.find_last_of(".") + 1).compare(".csv")) //Make sure it's a .csv file
    {
        cout << "Error: Charge CSV file is not a .csv file." << endl;
        return 1;
    }
    //Start reading through the file and adding amino acids to the list
    ifstream csvfile(chargefilepath);
    if (csvfile.is_open()) //Can we open the file?
    {
        cout << "Reading CSV file to obtain charges." << endl;
        string line;
        getline(csvfile, line); //Skip the header line
        int i = 0;
        while (getline(csvfile, line)) //Read each line through the file
        {
            
            auto linecontents = split(line,",");
            array<string, 2> entry;
            if (linecontents[chargecolumn].empty())
            {
                cout << "Warning: No charge for " << linecontents[residuecolumn] << ":" << linecontents[atomcolumn] << " found.  Setting to 0." << endl;
                entry = { linecontents[residuecolumn] + "-" + linecontents[atomcolumn], "0.0" };
                i++;
            }
            else
            {
                entry = { linecontents[residuecolumn] + "-" + linecontents[atomcolumn], linecontents[chargecolumn] };
            }
            chargedata.push_back(entry);
        }
    }
    else
    {
        cout << "Error: Unable to read " << chargefilepath << endl;
        return 1;
    }
    csvfile.close();
    return 0;
}