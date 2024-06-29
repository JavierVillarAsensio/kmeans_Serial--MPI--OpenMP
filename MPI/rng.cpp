#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <random>

#define K 5
#define ROWS 1096
#define COLS 715

using namespace std;

int main(){
    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<int> distribution(0, (ROWS * COLS) - 1);

    ofstream file;
    file.open("rng.txt");

    for(int i = 0; i < K; i++){
        int n = distribution(generator);
        file << n << ",";
    }

    file.close();
}