#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
using namespace std;

const int EMBED_DIM = 768;

void transpose_and_save(const string &infile, const string &outfile) {
    vector<vector<int>> mat(EMBED_DIM, vector<int>(EMBED_DIM));
    ifstream fin(infile);
    if (!fin.is_open()) { cerr << "Cannot open " << infile << endl; return; }

    // 讀入矩陣
    for (int i = 0; i < EMBED_DIM; i++) {
        string line; getline(fin, line);
        istringstream iss(line);
        for (int j = 0; j < EMBED_DIM; j++) {
            iss >> mat[i][j];
        }
    }
    fin.close();

    // 轉置並寫入
    ofstream fout(outfile);
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            fout << mat[j][i];
            if (j != EMBED_DIM - 1) fout << " ";
        }
        fout << "\n";
    }
    fout.close();
    cout << "Transposed matrix saved to " << outfile << endl;
}

int main() {
    transpose_and_save("Wq.txt", "Wq_transpose.txt");
    transpose_and_save("Wk.txt", "Wk_transpose.txt");
    transpose_and_save("Wv.txt", "Wv_transpose.txt");
    return 0;
}