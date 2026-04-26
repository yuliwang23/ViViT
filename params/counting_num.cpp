#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::string filename = "/home/undergraduate/rjchen25/ViViT/params/vivit_video_input_AfterConvAndEmbedding/sample_video_9_in.txt"; // file name
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "can't open the file" << std::endl;
        return 1;
    }

    float num;
    int count = 0;
    while (file >> num) {
        count++;
    }

    std::cout << "There are " << count << " number(float or int) in this file" << std::endl;

    file.close();
    return 0;
}

