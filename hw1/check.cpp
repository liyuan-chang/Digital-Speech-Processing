// **************************************************************************
//  File       [check.cpp]
//  Author     [Li-Yuan Chang]
//  Modify     [2020/10/30 Li-Yuan Chang]
// **************************************************************************

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

void help_message() {
    cout << "Usage: ./check <answer> <testing_result>" << endl;
    cout << "Example:" << endl;
    cout << "./check test_lbl.txt result.txt" << endl;
}

int main(int argc, char* argv[])
{
    if(argc != 3) {
       help_message();
       return 0;
    }

    vector<string> ans;
    vector<string> result;
    // read the input file
    string buffer;
    double prob;
    ifstream fin(argv[1]);
    while(getline(fin, buffer)){
        ans.push_back(buffer);
    }
    fin.close();
    fin.open(argv[2]);
    while(fin >> buffer >> prob){
        result.push_back(buffer);
    }
    fin.close();

    int count = 0;
    double accuracy = 0;
    for(int i = 0; i < ans.size(); ++i){
        if(ans[i] != result[i])
            ++count;
    }
    accuracy = 1 - double(count) / ans.size();
    cout << "Number of errors: " << count << endl;
    cout << "Accuracy: " << accuracy << endl;
    return 0;
}