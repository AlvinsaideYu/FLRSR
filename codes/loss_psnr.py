




import re
import pandas as pd

# 读取日志文件
log_file = r'D:\Wz_Project\HAUNet_RSISR\experiment\FSMamba4x4_UCMerced\log.txt'  # 修改为你的日志文件路径
l1_values = []
psnr_values = []

with open(log_file, 'r') as file:
    for line in file:
        # 提取L1值
        l1_match = re.search(r'L1:\s*([\d\.]+)', line)
        if l1_match:
            l1_values.append(float(l1_match.group(1)))

        # 提取PSNR值
        psnr_match = re.search(r'psnr:\s*([\d\.]+)', line)
        if psnr_match:
            psnr_values.append(float(psnr_match.group(1)))

# 检查长度是否一致
if len(l1_values) != len(psnr_values):
    print(f"警告：L1和PSNR的数量不一致！L1数量: {len(l1_values)}, PSNR数量: {len(psnr_values)}")

# 创建DataFrame并保存为Excel文件
df = pd.DataFrame({'L1': l1_values, 'PSNR': psnr_values})
excel_file =  r'D:\Wz_Project\HAUNet_RSISR\experiment\FSMamba4x4_UCMerced\output.xlsx' # 修改为你想要保存的Excel文件路径
df.to_excel(excel_file, index=False)

print("提取完成，结果已保存到Excel文件。")
