import numpy as np

def analyze_outputs(golden_file, hw_file):
    print(f"Loading files...")
    g_data = np.loadtxt(golden_file).reshape(197, 768).flatten()
    h_data = np.loadtxt(hw_file).reshape(197, 768).flatten()

    abs_diff = np.abs(g_data - h_data)
    
    print("-" * 50)
    print("Statistical Analysis:")
    print(f"Golden Data - Mean: {np.mean(g_data):.6f}, Std: {np.std(g_data):.6f}, Min: {np.min(g_data):.6f}, Max: {np.max(g_data):.6f}")
    print(f"HW Data     - Mean: {np.mean(h_data):.6f}, Std: {np.std(h_data):.6f}, Min: {np.min(h_data):.6f}, Max: {np.max(h_data):.6f}")
    print("-" * 50)
    
    print(f"Error Analysis:")
    print(f"Mean Absolute Error: {np.mean(abs_diff):.6f}")
    print(f"Max Absolute Error : {np.max(abs_diff):.6f}")
    print(f"Cosine Similarity  : {np.dot(g_data, h_data) / (np.linalg.norm(g_data) * np.linalg.norm(h_data)):.6f}")
    print("-" * 50)

    # 找出誤差大於 1.0 的數量
    large_errors = np.sum(abs_diff > 0.01)
    total_elements = g_data.size
    print(f"Elements with error > 0.01 : {large_errors} / {total_elements} ({large_errors/total_elements*100:.2f}%)")

    # 印出 Top 10 誤差，看有沒有正負號相反或是某一方全為 0 的情況
    print("\nTop 10 Largest Errors:")
    flat_indices = np.argsort(abs_diff)[::-1][:1000]
    for idx in flat_indices:
        print(f"  Index {idx:6d}: Golden = {g_data[idx]:10.6f}, HW = {h_data[idx]:10.6f} (Diff: {abs_diff[idx]:.6f})")

if __name__ == '__main__':
    analyze_outputs("gloden_output.txt", "Out_output.txt")