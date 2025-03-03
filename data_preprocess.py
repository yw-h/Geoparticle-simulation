import subprocess
import os

input_dir = "./result_compressedfield" # 假设模拟结果保存在 result_compressedfield 目录
output_dir = "./result_compressedfield/processed_results" # 预处理结果保存目录
os.makedirs(output_dir, exist_ok=True) # 创建输出目录

try:
    # 构建命令行命令来运行 raw_result_process1.1.py
    command = [
        "python",
        "raw_result_process1.1.py", # 假设 raw_result_process1.1.py 和 multi_sim_deepseek.py 在同一目录下
        input_dir,
        output_dir,
        "--flux_csv", './mageis_extended_fedu_mean.csv', # 假设 observed_flux.csv 在当前目录下，可以根据实际情况修改路径
        # "--batch_size", "10", # 可以根据需要调整参数
        "--process_type", "both"
    ]
    subprocess.run(command, check=True) # 运行预处理脚本
except Exception as e:
    print(f"Error during preprocessing execution: {e}")