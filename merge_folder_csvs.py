import pandas as pd
import os
import numpy as np

def merge_folder_csvs_to_base(folder_path, base_csv_path, output_csv_path="merged_all_stations_data.csv", time_col_name="time"):
    """
    将指定文件夹中所有CSV文件的内容合并到基准CSV文件中。
    除时间列外，所有其他列的名称将添加文件名前缀，并对缺失值进行线性插值。

    Args:
        folder_path (str): 包含要合并的CSV文件的文件夹路径。
        base_csv_path (str): 作为合并基准的CSV文件路径。
        output_csv_path (str): 合并后输出CSV文件的路径。
        time_col_name (str): 时间列的名称，默认为 'time'。
                              假设所有CSV文件（包括基准文件和文件夹中的文件）
                              都包含这个时间列，并且格式一致。
    """
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return
    if not os.path.exists(base_csv_path):
        print(f"错误：基准CSV文件 '{base_csv_path}' 不存在。")
        return

    # 1. 读取基准CSV文件
    try:
        base_df = pd.read_csv(base_csv_path)
        base_df[time_col_name] = pd.to_datetime(base_df[time_col_name])
        print(f"已加载基准文件：'{base_csv_path}'")
        print(f"基准文件包含 {len(base_df.columns) - 1} 个数据列。")
    except Exception as e:
        print(f"错误：加载或处理基准CSV文件 '{base_csv_path}' 失败。详细信息：{e}")
        return

    # 初始化用于合并的DataFrame，以基准文件为起点
    merged_df = base_df

    # 2. 遍历文件夹中的所有CSV文件
    csv_files_in_folder = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files_in_folder:
        print(f"警告：文件夹 '{folder_path}' 中没有找到CSV文件。只处理了基准文件。")
        # 如果没有其他CSV文件，只对基准文件进行插值（如果需要的话），然后保存
        if merged_df.isnull().any().any():
            print("基准文件包含缺失值，正在进行线性插值...")
            merged_df = merged_df.set_index(time_col_name)
            merged_df = merged_df.interpolate(method='linear', limit_direction='both')
            merged_df = merged_df.reset_index()
        merged_df.to_csv(output_csv_path, index=False)
        print(f"处理完成，结果已保存到 '{output_csv_path}'")
        return

    print(f"\n正在处理文件夹 '{folder_path}' 中的 {len(csv_files_in_folder)} 个CSV文件...")

    for filename in csv_files_in_folder:
        file_path = os.path.join(folder_path, filename)
        station_id = os.path.splitext(filename)[0] # 获取文件名作为前缀，例如 '41_200_114_700'

        try:
            df_station = pd.read_csv(file_path)
            # 转换时间列
            df_station[time_col_name] = pd.to_datetime(df_station[time_col_name])

            # 为除时间列之外的所有列添加前缀
            new_columns = {col: f"{station_id}_{col}" for col in df_station.columns if col != time_col_name}
            df_station = df_station.rename(columns=new_columns)

            # 将当前站点的DataFrame与总的合并DataFrame进行外连接
            # 使用 'time_col_name' 作为共同的键
            merged_df = pd.merge(merged_df, df_station, on=time_col_name, how='outer', suffixes=('_base', '_station'))
            print(f"已合并文件：'{filename}'。新列已添加前缀。")

        except Exception as e:
            print(f"警告：处理文件 '{filename}' 时出错。跳过此文件。详细信息：{e}")
            continue

    # 3. 排序所有数据以确保插值正确进行
    merged_df = merged_df.sort_values(by=time_col_name).reset_index(drop=True)

    # 4. 对所有非时间列的缺失值进行线性插值
    print("\n正在对合并后的数据进行线性插值...")
    # 先设置时间列为索引，方便插值
    merged_df = merged_df.set_index(time_col_name)
    merged_df = merged_df.interpolate(method='linear', limit_direction='both', axis=0) # axis=0表示按列插值
    merged_df = merged_df.reset_index() # 插值完成后再将时间列恢复为普通列

    # 5. 保存结果到新的CSV文件
    merged_df.to_csv(output_csv_path, index=False)
    print(f"\n所有文件合并与插值完成，结果已保存到 '{output_csv_path}'")

    # 打印一些合并后的数据概览
    print("\n合并后的数据前5行：")
    print(merged_df.head())
    print("\n合并后的数据信息：")
    merged_df.info()

# --- 使用示例 ---
if __name__ == "__main__":
    # --- 配置你的文件和文件夹路径 ---
    # 假设你的所有站点CSV文件都在 'station_data' 文件夹中
    # 假设你的基准CSV文件名为 'base_data.csv'
    # 请根据你的实际情况修改这些路径

    # 示例：创建一些模拟数据和文件夹以供测试
    print("正在创建模拟数据和文件夹用于演示...")
    # 创建文件夹
    test_folder = "station_data"
    os.makedirs(test_folder, exist_ok=True)

    # 模拟基准文件 base_data.csv
    base_data = {
        'time': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00', '2024-01-01 03:00:00'],
        'base_var_A': [10, 12, 15, 13],
        'base_var_B': [100, 105, 110, 102]
    }
    df_base = pd.DataFrame(base_data)
    df_base.to_csv("base_data.csv", index=False)
    print("模拟基准文件 'base_data.csv' 已创建。")

    # 模拟站点CSV文件
    # 站点 41_200_114_700.csv
    station1_data = {
        'time': ['2024-01-01 00:00:00', '2024-01-01 00:30:00', '2024-01-01 01:00:00', '2024-01-01 02:15:00'],
        'dewpoint_temperature_surface_2_metre': [2.5, 3.0, 3.8, 4.5],
        'relative_humidity_isobaric_1000': [80, 82, 85, 81]
    }
    df_station1 = pd.DataFrame(station1_data)
    df_station1.to_csv(os.path.join(test_folder, "41_200_114_700.csv"), index=False)
    print("模拟站点文件 'station_data/41_200_114_700.csv' 已创建。")

    # 站点 42_100_200_500.csv
    station2_data = {
        'time': ['2024-01-01 00:00:00', '2024-01-01 01:30:00', '2024-01-01 03:00:00'],
        'wind_speed_10_metre': [5.1, 6.2, 5.5],
        'air_temperature_2_metre': [18.0, 19.5, 18.8]
    }
    df_station2 = pd.DataFrame(station2_data)
    df_station2.to_csv(os.path.join(test_folder, "42_100_200_500.csv"), index=False)
    print("模拟站点文件 'station_data/42_100_200_500.csv' 已创建。")

    # 演示数据创建完毕
    print("-" * 30)

    # --- 调用函数执行合并 ---
    folder_of_csvs = test_folder            # 包含所有站点CSV文件的文件夹
    base_file = "base_data.csv"             # 你的基准CSV文件
    final_output_file = "final_merged_output.csv" # 最终输出文件名
    common_time_column = "time"             # 所有文件共同的时间列名

    merge_folder_csvs_to_base(folder_of_csvs, base_file, final_output_file, common_time_column)

    # 清理模拟数据 (可选)
    # import shutil
    # shutil.rmtree(test_folder)
    # os.remove(base_file)
    # os.remove(final_output_file)
    # print("\n已删除模拟数据和文件夹。")
