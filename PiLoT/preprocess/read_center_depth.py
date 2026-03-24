import os
import numpy as np

def generate_depth_txt(source_folder, output_txt_path, center_row=270, center_col=480):
    """
    读取文件夹中所有 npy 文件的中心点深度，并保存到 txt 文件中。
    
    格式: filename depth_value
    支持浮点数坐标（会四舍五入到最近的整数索引）
    """
    
    # 如果是浮点数坐标，转换为整数索引
    center_row = int(round(center_row))
    center_col = int(round(center_col))
    
    # 1. 检查源文件夹
    if not os.path.exists(source_folder):
        print(f"❌ 错误: 文件夹不存在 - {source_folder}")
        return

    # 2. 获取所有 .npy 文件并排序
    # 过滤出 .npy 文件
    files = [f for f in os.listdir(source_folder) if f.endswith('.npy')]
    
    # 关键步骤：按数字顺序排序 (例如确保 10_0.npy 在 2_0.npy 后面)
    # 假设文件名格式为 "数字_后缀.npy"，如 "0_0.npy"
    try:
        files.sort(key=lambda x: int(x.split('_')[0]))
    except:
        print("⚠️ 文件名不包含数字，将使用默认字母排序")
        files.sort()

    print(f"📂 找到 {len(files)} 个文件，准备处理...")

    valid_count = 0
    
    # 3. 打开 txt 文件准备写入
    with open(output_txt_path, 'w', encoding='utf-8') as f_out:
        for filename in files:
            file_path = os.path.join(source_folder, filename)
            
            try:
                # 加载数据
                data = np.load(file_path)
                
                # 处理维度 (C, H, W) -> (H, W)
                if data.ndim == 3:
                    data = data.squeeze(0)
                
                h, w = data.shape
                
                # 检查索引是否越界
                if 0 <= center_row < h and 0 <= center_col < w:
                    # 读取深度值
                    depth_val = float(data[center_row, center_col])
                    
                    # 写入一行: 文件名 深度值
                    # 使用 .4f 保留4位小数
                    line = f"{filename} {depth_val:.4f}\n"
                    f_out.write(line)
                    
                    valid_count += 1
                else:
                    # 如果越界，写入 -1 或者其他标记
                    print(f"⚠️ {filename}: 索引越界 ({h}x{w})")
                    f_out.write(f"{filename} -1.0\n")

            except Exception as e:
                print(f"❌ 读取 {filename} 失败: {e}")
                f_out.write(f"{filename} error\n")

    print("="*30)
    print(f"✅ 处理完成！")
    print(f"📄 结果已保存至: {output_txt_path}")
    print(f"📊 成功提取: {valid_count}/{len(files)}")
    print("="*30)

if __name__ == "__main__":
    # ================= 配置区域 =================
    targets = [
        "switzerland_seq4@8@foggy@intensity1@100",
        "switzerland_seq4@8@foggy@intensity1@200",
        "switzerland_seq4@8@foggy@intensity1@300",
        "switzerland_seq4@8@foggy@intensity1@400",
        "switzerland_seq4@8@foggy@intensity1@500"
    ]
    for target in targets:
        NPY_FOLDER = f"/media/amax/PS2000/depth/{target}"
        OUTPUT_TXT = f"/media/amax/PS2000/depth/{target}.txt"
        ROW_IDX = 960
        COL_IDX = 540
        generate_depth_txt(NPY_FOLDER, OUTPUT_TXT, ROW_IDX, COL_IDX)

    # # 1. npy 文件夹路径 (请修改这里)
    # NPY_FOLDER = "/media/amax/PS2000/depth"
    
    # # 2. 输出 txt 文件路径 (请修改这里)
    # OUTPUT_TXT = "/media/amax/PS2000/depth/DJI_20251221132525_0004_V.txt"
    
    # # 3. 中心点坐标 (Row/Height/Y, Col/Width/X)
    # # 支持浮点数，程序会自动四舍五入到最近的整数索引
    # # 960x540 -> row=270, col=480
    # # 512x512 -> row=256, col=256
    # ROW_IDX = 960
    # COL_IDX = 540
    
    # # ===========================================
    
    # generate_depth_txt(NPY_FOLDER, OUTPUT_TXT, ROW_IDX, COL_IDX)