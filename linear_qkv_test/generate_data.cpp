/*#include <iostream>
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
}*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
using namespace std;

constexpr int SEQ_LEN = 14*14 + 1;
constexpr int EMBED_DIM = 768;

// 隨機整數
int rand_int()
{
    return rand() % 5 + 1;
}

// 產生 Identity matrix
void generate_identity_matrix(const char* filename, int size)
{
    ofstream fout(filename);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == j)
                fout << 1 << " ";
            else
                fout << 0 << " ";
        }
        fout << endl;
    }

    fout.close();
    cout << filename << " generated (Identity " << size << "x" << size << ")" << endl;
}

// 產生 random matrix
void generate_random_matrix(const char* filename, int rows, int cols)
{
    ofstream fout(filename);

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

// 產生 one-hot matrix
void generate_onehot_matrix(const char* filename, int rows, int cols)
{
    ofstream fout(filename);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (j == (i % cols))
                fout << 1 << " ";
            else
                fout << 0 << " ";
        }
        fout << endl;
    }

    fout.close();
    cout << filename << " generated (one-hot)" << endl;
}

// 產生全1 matrix
void generate_all1_matrix(const char* filename, int rows, int cols)
{
    ofstream fout(filename);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fout << 1 << " ";
        }
        fout << endl;
    }

    fout.close();
    cout << filename << " generated (all 1)" << endl;
}

// 產生 bias
void generate_bias(const char* filename, int size)
{
    ofstream fout(filename);

    for (int i = 0; i < size; i++)
    {
        // fout << i << endl;
        fout << 0 << endl;
    }

    fout.close();
    cout << filename << " generated (bias pattern)" << endl;
}

int rand_weight()
{
    return rand() % 11 - 5;   // -5 ~ 5
}

void generate_Wqkv_matrix(const char* filename, int rows, int cols)
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
            fout << rand_weight() << " ";
        }
        fout << endl;
    }

    fout.close();

    cout << filename << " generated (" << rows << "x" << cols << ")" << endl;
}

int main()
{
    srand(time(0));

    // X
    //generate_random_matrix("X.txt", SEQ_LEN, EMBED_DIM);
    generate_onehot_matrix("X.txt", SEQ_LEN, EMBED_DIM);

    // 測試1：Identity weight
    /*generate_identity_matrix("Wq.txt", EMBED_DIM);
    generate_identity_matrix("Wk.txt", EMBED_DIM);
    generate_identity_matrix("Wv.txt", EMBED_DIM);*/

    // 測試2：One-Hot weight
    generate_Wqkv_matrix("Wq.txt", EMBED_DIM, EMBED_DIM);
    generate_Wqkv_matrix("Wk.txt", EMBED_DIM, EMBED_DIM);
    generate_Wqkv_matrix("Wv.txt", EMBED_DIM, EMBED_DIM);

    // bias
    generate_bias("bq.txt", EMBED_DIM);
    generate_bias("bk.txt", EMBED_DIM);
    generate_bias("bv.txt", EMBED_DIM);

    cout << endl;
    cout << "All test data generated successfully." << endl;

    return 0;
}