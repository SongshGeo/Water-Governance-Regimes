#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import numpy as np
import pandas as pd

from regimes_yrb.tools.statistic import pettitt_change_points, zscore

# 设置探索断点的显著性阈值
p_shr = 0.0005

# 第二种 Z-score 标准化方法，对断点无影响
# def zscore(x):
#     average = x.mean(numeric_only=True)
#     stddev = x.std()
#     return (x-average)/stddev


# # 第三种 [-1, 1] 之间的标准化，也对断点无影响
# def zscore(x):
#     return (x-x.mean(numeric_only=True))/(x.max()-x.min())


def integrated_water_governance_index(
    scarcity: np.ndarray,
    priority: np.ndarray,
    allocation: np.ndarray,
    norm: str = "log_zscore",
) -> pd.DataFrame:
    """计算综合水治理指数 Integrated Water Governance Index."""
    # 计算 iwgi 指数
    data = pd.DataFrame({"S": scarcity, "P": priority, "A": allocation})
    if norm == "log_zscore":
        norm_data = data.apply(np.log).apply(zscore)
    else:
        raise KeyError("'norm' must be 'log_zscore' or 'zscore'")
    norm_data["IWGI"] = norm_data.sum(axis=1) / 3
    return norm_data


def calc_contribution(iwgi, threshold, span=5):
    """计算每个阶段的贡献度"""
    priority = iwgi["P"]
    allocation = iwgi["A"]
    scarcity = iwgi["S"]

    dims = ["P", "A", "S"]  # 数据集字典的键
    contribution = {index: [] for index in dims}  # 用于储存贡献率计算结果的字典
    periods = []  # 用于储存时间段划分的空列表
    changes = {}  # 储存每个时间段的变化

    # 寻找断点
    breakpoints = pettitt_change_points(iwgi["IWGI"], p_shr=threshold)
    change_points = breakpoints.copy()
    change_points.extend([iwgi.index.max(), iwgi.index.min()])  # 为断点增加头尾
    check_points = sorted(change_points.copy())  # 排序好的断点列表作为检查点

    # 循环，每两个断点之间算一个阶段，计算每个阶段不同指数的贡献率。
    for i in range(len(check_points) - 1):
        j = i + 1
        start_year = check_points[i]
        end_year = check_points[j]
        if end_year in breakpoints:
            end_year = end_year - 1
        period = f"P{j}: {start_year}-{end_year}"
        periods.append(period)  # 每个阶段的标准化格式

        # 计算 IWGI 总的变化
        changes[period] = abs(
            iwgi["IWGI"]
            .loc[end_year - span : end_year]
            .mean(numeric_only=True)
            - iwgi["IWGI"]
            .loc[start_year : start_year + span]
            .mean(numeric_only=True)
        )
        change_iwgi = iwgi["IWGI"].loc[end_year - span : end_year].mean(
            numeric_only=True
        ) - iwgi["IWGI"].loc[start_year : start_year + span].mean(
            numeric_only=True
        )

        # 再计算三个分指标的变化
        change_p = priority[end_year] - priority[start_year]
        change_a = allocation[end_year] - allocation[start_year]
        change_s = scarcity[start_year] - scarcity[end_year]

        # 计算变化的贡献率并存入结果字典
        contribution["P"].append(100 * change_p / abs(change_iwgi))
        contribution["A"].append(100 * change_a / abs(change_iwgi))
        contribution["S"].append(100 * change_s / abs(change_iwgi))

    # 将结果转化为 DataFrame 并返回
    contribution = pd.DataFrame(contribution, index=periods)
    # 变化贡献
    changes_contribution = pd.DataFrame(columns=dims)
    for col in contribution:
        changes_contribution[col] = (
            contribution[col] * pd.Series(changes) / 100
        )
    return changes_contribution
