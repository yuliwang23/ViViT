#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#define ROWS 197
#define COLS 768
#define THRESHOLD 1e-3   // 容許誤差

bool compare_files(const char* file1, const char* file2)
{
    ifstream f1(file1);
    ifstream f2(file2);

    if (!f1.is_open() || !f2.is_open())
    {
        cout << "Error opening file: " << file1 << " or " << file2 << endl;
        return false;
    }

    float a, b;
    bool same = true;

    for(int i = 0; i < ROWS; i++)
    {
        for(int j = 0; j < COLS; j++)
        {
            f1 >> a;
            f2 >> b;

            if (fabs(a - b) > THRESHOLD)
            {
                cout << "Mismatch at [" << i << "][" << j << "] "
                     << " : " << a << " vs " << b << endl;

                same = false;
            }
        }
    }

    if(same)
        cout << "Files are identical: " << file1 << " and " << file2 << endl;
    else
        cout << "Files have differences: " << file1 << " and " << file2 << endl;

    f1.close();
    f2.close();

    return same;
}

int main()
{
    cout << "Comparing Q..." << endl;
    compare_files("Q_output_sw.txt", "Q_output_hw.txt");

    cout << endl;

    cout << "Comparing K..." << endl;
    compare_files("K_output_sw.txt", "K_output_hw.txt");

    cout << endl;

    cout << "Comparing V..." << endl;
    compare_files("V_output_sw.txt", "V_output_hw.txt");

    return 0;
}