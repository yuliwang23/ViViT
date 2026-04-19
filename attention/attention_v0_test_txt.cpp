#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

// 根據你的標頭檔定義常數
// SEQ_LEN = 14 * 14 + 1 = 197
// HEAD_DIM = 768 / 12 = 64
const int SEQ_LEN = 197;
const int HEAD_DIM = 64;

void generate_input_file(const char* filename, int rows, int cols) {
    std::ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        std::cerr << "無法開啟檔案: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 生成 -10 到 10 之間的隨機整數，模擬你的測試環境
            int val = (rand() % 21) - 10;
            outfile << val;
            
            // 元素間用空格隔開，行尾不留空格
            if (j < cols - 1) {
                outfile << " ";
            }
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "已成功生成檔案: " << filename << " (大小: " << rows << "x" << cols << ")" << std::endl;
}

int main() {
    // 設定隨機種子以確保每次生成或測試的一致性 (若需固定種子可改用 srand(0))
    srand(static_cast<unsigned int>(time(0)));

    std::cout << "正在為 HLS Attention 模組生成測試資料..." << std::endl;
    std::cout << "SEQ_LEN: " << SEQ_LEN << ", HEAD_DIM: " << HEAD_DIM << "\n" << std::endl;

    // 分別生成 Q, K, V 三個檔案
    generate_input_file("input_q.txt", SEQ_LEN, HEAD_DIM);
    generate_input_file("input_k.txt", SEQ_LEN, HEAD_DIM);
    generate_input_file("input_v.txt", SEQ_LEN, HEAD_DIM);

    std::cout << "\n所有輸入檔案已準備就緒。" << std::endl;

    return 0;
}