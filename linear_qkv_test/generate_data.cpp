#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

constexpr int SEQ_LEN = 14*14 + 1;
constexpr int EMBED_DIM = 768;

int rand_int()
{
    return rand() % 5 + 1;   // 產生 1~5
}

// 產生 matrix
void generate_matrix(const char* filename, int rows, int cols)
{
    ofstream fout(filename);

    if (!fout.is_open())
    {
        cout << "Cannot open " << filename << endl;
        return;
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fout << rand_int() << " ";
        }
        fout << endl;
    }

    fout.close();

    cout << filename << " generated (" << rows << "x" << cols << ")" << endl;
}

// 產生 vector
void generate_vector(const char* filename, int size)
{
    ofstream fout(filename);

    if (!fout.is_open())
    {
        cout << "Cannot open " << filename << endl;
        return;
    }

    for (int i = 0; i < size; i++)
    {
        fout << rand_int() << endl;
    }

    fout.close();

    cout << filename << " generated (" << size << ")" << endl;
}

int main()
{
    srand(time(0));

    // X
    generate_matrix("X.txt", SEQ_LEN, EMBED_DIM);

    // Wq Wk Wv
    generate_matrix("Wq.txt", EMBED_DIM, EMBED_DIM);
    generate_matrix("Wk.txt", EMBED_DIM, EMBED_DIM);
    generate_matrix("Wv.txt", EMBED_DIM, EMBED_DIM);

    // bias
    generate_vector("bq.txt", EMBED_DIM);
    generate_vector("bk.txt", EMBED_DIM);
    generate_vector("bv.txt", EMBED_DIM);

    cout << endl;
    cout << "All test data generated successfully." << endl;

    return 0;
}