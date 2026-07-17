#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算理论功率(theoretical_power)。

公式: theoretical_power = air_density * (ws_100m ^ 3)
其中 ws_100m 由风速替代属性列 component_of_wind_wind_component_surface_100_metre_* 提供。

用法:
    python tools/compute_theoretical_power.py \
        --air-density air_density.csv \
        --wind-data weather.csv \
        --wind-col component_of_wind_wind_component_surface_100_metre_41_400_114_900 \
        --output theoretical_power.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def load_csv(path: str, required_cols: list) -> pd.DataFrame:
    """读取 CSV 并校验必需列。

    Args:
        path: CSV 文件路径。
        required_cols: 必需列名列表。

    Returns:
        DataFrame。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺少必需列或类型错误。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    df = pd.read_csv(path)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"文件 {path} 缺少必需列: {missing}，当前列: {list(df.columns)}")

    for col in required_cols:
        if col == "time":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"列 '{col}' 不是数值类型: {df[col].dtype}")

    return df


def merge_on_time(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """基于 time 列合并两个 DataFrame。

    Args:
        df_a: 含 time 列的 DataFrame。
        df_b: 含 time 列的 DataFrame。

    Returns:
        合并后的 DataFrame。

    Raises:
        ValueError: 合并后行数不匹配。
    """
    df_a["time"] = pd.to_datetime(df_a["time"])
    df_b["time"] = pd.to_datetime(df_b["time"])

    merged = pd.merge(df_a, df_b, on="time", how="inner")

    if len(merged) == 0:
        raise ValueError("两个文件的时间列无交集，请检查数据")

    if len(merged) != len(df_a):
        print(f"[警告] air_density 文件有 {len(df_a)} 行，合并后剩 {len(merged)} 行")

    return merged


def compute_theoretical_power(air_density: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    """计算理论功率。

    公式: theoretical_power = air_density * (ws_100m ^ 3)

    Args:
        air_density: 空气密度数组。
        wind_speed: 100米高度风速数组。

    Returns:
        理论功率数组，无效值设为 NaN。
    """
    result = np.full_like(air_density, np.nan, dtype=np.float64)

    valid_mask = (~np.isnan(air_density)) & (~np.isnan(wind_speed)) & (wind_speed >= 0)
    result[valid_mask] = air_density[valid_mask] * np.power(wind_speed[valid_mask], 3)

    invalid_count = int(np.sum(~valid_mask))
    if invalid_count > 0:
        print(f"[警告] {invalid_count} 行因 NaN 或风速<0 无法计算，设为 NaN")

    return result


def main():
    parser = argparse.ArgumentParser(description="计算理论功率 theoretical_power")
    parser.add_argument("--air-density", required=True,
                        help="含 air_density 列的 CSV 文件路径（由 compute_air_density.py 生成）")
    parser.add_argument("--wind-data", required=True,
                        help="含风速替代属性列的 CSV 文件路径")
    parser.add_argument("--wind-col",
                        default="component_of_wind_wind_component_surface_100_metre_41_400_114_900",
                        help="风速替代列名（默认: component_of_wind_wind_component_surface_100_metre_41_400_114_900）")
    parser.add_argument("--output", default=None,
                        help="输出 CSV 文件路径（默认: theoretical_power.csv）")
    args = parser.parse_args()

    # 1. 读取空气密度数据
    df_ad = load_csv(args.air_density, ["time", "air_density"])
    print(f"空气密度: {len(df_ad)} 行")

    # 2. 读取风速数据
    df_wind = load_csv(args.wind_data, ["time", args.wind_col])
    print(f"风速数据: {len(df_wind)} 行，列: {list(df_wind.columns)}")

    # 3. 基于 time 合并
    merged = merge_on_time(df_ad[["time", "air_density"]], df_wind[["time", args.wind_col]])
    print(f"合并后: {len(merged)} 行")

    # 4. 计算理论功率
    air_density = merged["air_density"].to_numpy(dtype=np.float64)
    wind_speed = merged[args.wind_col].to_numpy(dtype=np.float64)
    theoretical_power = compute_theoretical_power(air_density, wind_speed)

    # 5. 组装结果
    result = pd.DataFrame()
    result["time"] = merged["time"]
    result["air_density"] = air_density
    result["theoretical_power"] = theoretical_power

    # 6. 保存
    output = args.output or "theoretical_power.csv"
    result.to_csv(output, index=False)
    print(f"已输出: {output}（{len(result)} 行）")


if __name__ == "__main__":
    main()
