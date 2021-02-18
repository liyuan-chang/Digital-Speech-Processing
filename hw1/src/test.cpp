// **************************************************************************
//  File       [test.cpp]
//  Author     [Li-Yuan Chang]
//  Modify     [2020/10/30 Li-Yuan Chang]
// **************************************************************************

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../inc/hmm.h"
#define N 6             // N: number of states =  number of observations ; T: time interval
#define T 50

using namespace std;

void help_message() {
    cout << "Usage: ./test <models_list_path> <seq_path> <output_result_path>" << endl;
    cout << "Example:" << endl;
    cout << "./test modellist.txt data/test_seq.txt result.txt" << endl;
}

int check_ot(const vector<string>&, int, int);

double Viterbi(const HMM&, const vector<string>&, int);

int main(int argc, char* argv[])
{
    if(argc != 4) {
       help_message();
       return 0;
    }

    HMM hmms[5];
    vector<string> model_name;
    vector<string> seq;         // testing sequences
    // read the input file
    string buffer;
    ifstream fin(argv[1]);
    while(getline(fin, buffer)){
        model_name.push_back(buffer);
    }
    fin.close();
    load_models(argv[1], hmms, 5);
    fin.open(argv[2]);
    while(getline(fin, buffer)){
        seq.push_back(buffer);
    }
    fin.close();
    
    // open the output file
    ofstream fout(argv[3]);

    ////// the testing part //////
    int seq_num = seq.size();
    int max_model = 0;
    double max_prob = 0;
    for(int n = 0; n < seq_num; ++n){
        for(int i = 0; i < 5; ++i){
            double prob = Viterbi(hmms[i], seq, n);
            if(prob > max_prob){
                max_prob = prob;
                max_model = i;
            }
        }
        // write to the output file
        fout << model_name[max_model] << ' ' << max_prob << endl;
        max_model = 0;
        max_prob = 0;
    }

    fout.close();
    return 0;
}

int check_ot(const vector<string>& seq, int seq_cnt, int t){
    int observe;
    switch(seq[seq_cnt][t]){
        case 'A':    observe = 0; break;
        case 'B':    observe = 1; break;
        case 'C':    observe = 2; break;
        case 'D':    observe = 3; break;
        case 'E':    observe = 4; break;
        case 'F':    observe = 5; break;
    }
    return observe;
}

double Viterbi(const HMM& hmm, const vector<string>& seq, int seq_cnt){
    double delta[N][T];
    double max = 0;
    double p = 0;
    for(int i = 0; i < N; ++i){
        int o1 = check_ot(seq, seq_cnt, 0);           // check o_1 = A, B, C, D, E, F
        delta[i][0] = hmm.initial[i] * hmm.observation[o1][i];
    }
    for(int t = 1; t < T; ++t){     // t: time ; j: current_state ; i: previous_state
        for(int j = 0; j < N; ++j){
            for(int i = 0; i < N; ++i){
                if(delta[i][t-1] * hmm.transition[i][j] > max)
                    max = delta[i][t-1] * hmm.transition[i][j];
            }
            int ot = check_ot(seq, seq_cnt, t);           // check o_t = A, B, C, D, E, F
            delta[j][t] = max * hmm.observation[ot][j];
            max = 0;
        }
    }
    for(int i = 0; i < N; ++i){
        if(delta[i][T-1] > p)
            p = delta[i][T-1];
    }
    return p;
}