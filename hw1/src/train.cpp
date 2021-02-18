// **************************************************************************
//  File       [train.cpp]
//  Author     [Li-Yuan Chang]
//  Modify     [2020/10/31 Li-Yuan Chang]
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
    cout << "Usage: ./train <iter> <model_init_path> <seq_path> <output_model_path>" << endl;
    cout << "Example:" << endl;
    cout << "./train 100 model_init.txt data/train_seq_01.txt model_01.txt" << endl;
}

int check_ot(const vector<string>&, int, int);

struct Epsilon
{
    double e[N][N];
};

void train(const HMM&, const vector<string>&, double [][T], Epsilon [], int);
void init_hmm(HMM&, char*, int, int);

int main(int argc, char* argv[])
{
    if(argc != 5) {
       help_message();
       return 0;
    }

    HMM hmm;

    // read the input file
    loadHMM(&hmm, argv[2]);
    string buffer;
    ifstream fin(argv[3]);
    vector<string> seq;     // training sequences
    while (getline(fin, buffer)){
        seq.push_back(buffer);
    }
    fin.close();
    dumpHMM(open_or_die("a.txt", "w"), &hmm);
    ////// the training part //////      
    int iters = atoi(argv[1]);
    double gamma[N][T];
    Epsilon epsilon[T-1];
    HMM new_hmm;
    int seq_num = seq.size();
    double sum = 0;
    double gamma_sum_tran[N];
    double gamma_sum_observe[N];
    for(int iter = 0; iter < iters; ++iter){
        // reset
        init_hmm(new_hmm, argv[4], N, N);
        for(int i = 0; i < N; ++i){
            gamma_sum_tran[i] = 0;
            gamma_sum_observe[i] = 0;
        }
        // train
        for(int n = 0; n < seq_num; ++n){
            train(hmm, seq, gamma, epsilon, n);
            // update the numerator of hmm model
            for(int i = 0; i < N; ++i)
                new_hmm.initial[i] += gamma[i][0];
            for(int i = 0; i < N; ++i){
                for(int j = 0; j < N; ++j){
                    for(int t = 0; t < T-1; ++t)
                        sum += epsilon[t].e[i][j];
                    new_hmm.transition[i][j] += sum;
                    sum = 0;
                }
            }
            for(int i = 0; i < N; ++i){
                for(int k = 0; k < N; ++k){
                    for(int t = 0; t < T-1; ++t){
                        if(check_ot(seq, n, t) == k)
                            sum += gamma[i][t];
                    }
                    new_hmm.observation[k][i] += sum;
                    sum = 0;
                }
            }
            // calculate the denominator of hmm model
            for(int i = 0; i < N; ++i){
                for(int t = 0; t < T-1; ++t){
                    gamma_sum_tran[i] += gamma[i][t];
                    gamma_sum_observe[i] += gamma[i][t];
                }
                gamma_sum_observe[i] += gamma[i][T-1];
            }
        }
        // update the denominator of hmm model
        for(int i = 0; i < N; ++i)
            new_hmm.initial[i] /= seq_num;
        for(int i = 0; i < N; ++i){
            for(int j = 0; j < N; ++j)
                new_hmm.transition[i][j] /= gamma_sum_tran[i];
        }
        for(int i = 0; i < N; ++i){
            for(int k = 0; k < N; ++k)
                new_hmm.observation[k][i] /= gamma_sum_observe[i];
        }
        hmm = new_hmm;
    }
    // write the output file
    dumpHMM(open_or_die(argv[4], "w"), &hmm);
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

void train(const HMM& hmm, const vector<string>& seq, double gamma[][T], Epsilon epsilon[], int seq_cnt){
    // forward algorithm
    double a[N][T];     // alpha
    double sum = 0;
    for(int i = 0; i < N; ++i){
        int o1 = check_ot(seq, seq_cnt, 0);           // check o_1 = A, B, C, D, E, F
        a[i][0] = hmm.initial[i] * hmm.observation[o1][i];
    }
    for(int t = 1; t < T; ++t){     // t: time ; j: state
        int ot = check_ot(seq, seq_cnt, t);           // check o_t = A, B, C, D, E, F
        for(int j = 0; j < N; ++j){
            for(int k = 0; k < N; ++k)
                sum += a[k][t-1] * hmm.transition[k][j];
            a[j][t] = sum * hmm.observation[ot][j];
            sum = 0;
        }
    }
    // backward algorithm
    double b[N][T];     // beta
    sum = 0;
    for(int i = 0; i < N; ++i)
        b[i][T-1] = 1;
    for(int t = T-2; t >= 0; --t){     // t: time ; i: state
        int ot1 = check_ot(seq, seq_cnt, t+1);           // check o_t+1 = A, B, C, D, E, F
        for(int i = 0; i < N; ++i){
            for(int j = 0; j < N; ++j)
                sum += b[j][t+1] * hmm.transition[i][j] * hmm.observation[ot1][j];
            b[i][t] = sum;
            sum = 0;
        }
    }
    // calculate gamma
    // gamma[N][T];
    sum = 0;
    for(int t = 0; t < T; ++t){     // t: time ; i: state
        for(int i = 0; i < N; ++i){
            gamma[i][t] = a[i][t] * b[i][t];
            sum += gamma[i][t];
        }
        for(int i = 0; i < N; ++i)
            gamma[i][t] /= sum;
        sum = 0;
    }
    // calculate epsilon
    // epsilon[T-1];
    sum = 0;
    for(int t = 0; t < T-1; ++t){     // t: time ; i: state; j: next_state
        int ot1 = check_ot(seq, seq_cnt, t+1);           // check o_t+1 = A, B, C, D, E, F
        for(int i = 0; i < N; ++i){
            for(int j = 0; j < N; ++j){
                epsilon[t].e[i][j] = a[i][t] * hmm.transition[i][j] * hmm.observation[ot1][j] * b[j][t+1];
                sum += epsilon[t].e[i][j];
            }
        }
        for(int i = 0; i < N; ++i){
            for(int j = 0; j < N; ++j)
                epsilon[t].e[i][j] /= sum;
        }
        sum = 0;
    }
}
void init_hmm(HMM& hmm, char *model_name, int state_num, int observ_num){
    hmm.model_name = model_name;
    hmm.state_num = state_num;
    hmm.observ_num = observ_num;
    for(int i = 0; i < N; ++i)
        hmm.initial[i] = 0;
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            hmm.transition[i][j] = 0;
            hmm.observation[i][j] = 0;
        }
    }
}