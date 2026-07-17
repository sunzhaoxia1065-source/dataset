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


def normalize_time(time_series: pd.Series) -> pd.Series:
    """将各种时间格式统一解析为 datetime。

    支持的格式:
      - YYYYMMDDHHMM (如 202401030000)
      - YYYY/M/D HH:MM:SS (如 2024/1/3 00:00:00)
      - YYYY-MM-DD HH:MM:SS (如 2024-01-03 00:00:00)
      - 其他 pandas 可自动识别的格式

    Args:
        time_series: 原始时间列。

    Returns:
        解析后的 datetime Series。
    """
    # 先尝试直接解析
    parsed = pd.to_datetime(time_series, errors="coerce")
    if parsed.isna().all():
        # 全部失败，尝试作为 YYYYMMDDHHMM 数字字符串解析
        try:
            parsed = pd.to_datetime(time_series.astype(str), format="%Y%m%d%H%M", errors="coerce")
        except Exception:
            pass

    # 仍有无法解析的，逐行尝试多种格式
    if parsed.isna().any():
        still_na = parsed.isna()
        for idx in still_na[still_na].index:
            val = str(time_series.iloc[idx]).strip()
            for fmt in ["%Y%m%d%H%M", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                        "%Y/%m/%d %H:%M", "%Y-%m-%d", "%Y%m%d"]:
                try:
                    parsed.iloc[idx] = pd.to_datetime(val, format=fmt)
                    break
                except (ValueError, TypeError):
                    continue

    if parsed.isna().all():
        raise ValueError(f"无法解析时间列，示例值: {time_series.head(3).tolist()}")

    # 截断到分钟精度，消除秒/毫秒差异
    parsed = parsed.dt.floor("min")

    na_count = parsed.isna().sum()
    if na_count > 0:
        print(f"[警告] {na_count} 个时间值无法解析")

    return parsed


def merge_on_time(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """基于 time 列合并两个 DataFrame，自动处理格式差异和时间粒度不对齐。

    如果精确匹配失败，会对两个时间列按分钟取整后重试；
    若仍无交集，则采用最近时间匹配（tolerance 可配置）。

    Args:
        df_a: 含 time 列的 DataFrame。
        df_b: 含 time 列的 DataFrame。

    Returns:
        合并后的 DataFrame。

    Raises:
        ValueError: 两个文件的时间列无交集。
    """
    df_a = df_a.copy()
    df_b = df_b.copy()

    # 统一解析时间格式
    df_a["time"] = normalize_time(df_a["time"])
    df_b["time"] = normalize_time(df_b["time"])

    print(f"  时间范围 A: {df_a['time'].min()} ~ {df_a['time'].max()} ({len(df_a)} 行)")
    print(f"  时间范围 B: {df_b['time'].min()} ~ {df_b['time'].max()} ({len(df_b)} 行)")

    # 尝试精确匹配
    merged = pd.merge(df_a, df_b, on="time", how="inner")

    if len(merged) > 0:
        print(f"  精确匹配: {len(merged)} 行")
        if len(merged) != len(df_a):
            print(f"[警告] air_density 文件有 {len(df_a)} 行，合并后剩 {len(merged)} 行")
        return merged

    # 精确匹配失败，尝试 merge_asof 最近时间匹配
    print("[提示] 精确匹配无结果，尝试最近时间匹配（容差15分钟）...")

    df_a = df_a.sort_values("time").reset_index(drop=True)
    df_b = df_b.sort_values("time").reset_index(drop=True)

    merged = pd.merge_asof(df_a, df_b, on="time", direction="nearest", tolerance=pd.Timedelta("15min"))

    # 去除未匹配的行（wind 列为 NaN）
    merged = merged.dropna(subset=[c for c in merged.columns if c != "time" and c != "air_density"])

    if len(merged) == 0:
        raise ValueError(
            "两个文件的时间列无交集（容差15分钟内也无匹配）。\n"
            f"  A 范围: {df_a['time'].min()} ~ {df_a['time'].max()}\n"
            f"  B 范围: {df_b['time'].min()} ~ {df_b['time'].max()}\n"
            "请检查两个文件的时间范围是否有重叠，或使用 merge_csv.py 先统一时间格式。"
        )

    print(f"  最近时间匹配: {len(merged)} 行")
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
