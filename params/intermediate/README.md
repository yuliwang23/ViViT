# ViViT 推論中間值紀錄 (Intermediate Inference Values)

本目錄存放了 ViViT 模型在進行推論（Inference）時，各層網路之間產生的中間計算結果（Intermediate Values）。這些數據可用於模型驗證、量化分析（Quantization Analysis）或偵錯。

---

## 目錄結構說明

* **`sample_video_0_out.txt` ~ `sample_video_9_out.txt`**
  共計 10 個樣本的推論中間值數據。每個檔案紀錄了該樣本在推論流中的數值變化。

* **`encoder_block.txt`**
  紀錄了各個數值標籤（Label）與其對應模型層級（Layer）位置的映射關係。

---

## 如何識別數值對應的層級位置

若要確認檔案中的數值屬於哪兩個層（Layer）之間的中間值，請參照原始碼中 `self.sow` 函數的實作邏輯：

### 查詢方式
在程式碼中搜尋以下格式：
> `self.sow('intermediate', {LABEL}, x)` 或 `y`

其中：
- **`{LABEL}`**：對應到檔案中紀錄的名稱（標籤）。
- **`x` (或 `y`)**：該標籤所紀錄的實際數值。

透過查看 `self.sow` 在代碼中插入的位置，即可精確判斷該標籤（Label）代表的是哪兩個運算層級之間的輸出結果。

---

## 數據規格
* **數量**：10 個測試樣本。
* **格式**：純文本格式，數值以浮點數紀錄。
* **建議**：讀取數據時，建議搭配 `numpy` 進行解析以方便與模型輸出進行對比分析。