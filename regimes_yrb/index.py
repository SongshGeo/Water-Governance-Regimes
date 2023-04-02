#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import numpy as np
import pandas as pd
from scipy import stats

from regimes_yrb.tools.statistic import get_optimal_fit_linear, pettitt_changes, zscore


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
        results = data.apply(np.log).apply(zscore)
    else:
        raise KeyError("'norm' must be 'log_zscore' or 'zscore'")
    results["IWGI"] = results.sum(axis=1) / 3
    breakpoints = pettitt_changes(results["IWGI"])
    results.loc[results.index < breakpoints[0], "stage"] = "P1"
    results.loc[
        (breakpoints[0] <= results.index) & (results.index < breakpoints[1]),
        "stage",
    ] = "P2"
    results.loc[breakpoints[1] <= results.index, "stage"] = "P3"
    return results


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
    breakpoints = pettitt_changes(iwgi["IWGI"], p_shr=threshold)
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
            iwgi["IWGI"].loc[end_year - span : end_year].mean(numeric_only=True)
            - iwgi["IWGI"].loc[start_year : start_year + span].mean(numeric_only=True)
        )
        change_iwgi = iwgi["IWGI"].loc[end_year - span : end_year].mean(
            numeric_only=True
        ) - iwgi["IWGI"].loc[start_year : start_year + span].mean(numeric_only=True)

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
        changes_contribution[col] = contribution[col] * pd.Series(changes) / 100
    return changes_contribution


# BREAK_POINTS = pettitt_change_points(iwgi["IWGI"], p_shr=threshold)


def calc_index_correlation(iwgi):
    """计算IWGI与三个维度指标在不同阶段的相关性"""
    # TODO refactor this
    k_results = pd.DataFrame(index=["P1", "P2", "P3"])
    # b_results = pd.DataFrame(index=["P1", "P2", "P3"])
    corr_results = pd.DataFrame(index=["P1", "P2", "P3"])
    p_results = pd.DataFrame(index=["P1", "P2", "P3"])
    breakpoints = [1965, 1978, 2001, 2013]

    for i, yr_start in enumerate(breakpoints[:3]):
        yr_end = breakpoints[i + 1]
        temp_data = iwgi.loc[yr_start:yr_end, ["S", "P", "A", "IWGI"]]
        for col in iwgi.iloc[:, :3]:
            x, y = temp_data.index, temp_data[col].values
            _, k, _ = get_optimal_fit_linear(x, y)
            k_results.loc[f"P{i+1}", col] = k
            if col != "IWGI":
                corr, p_val = stats.spearmanr(
                    temp_data[col].values, temp_data["IWGI"].values
                )
                corr_results.loc[f"P{i+1}", col] = corr
                p_results.loc[f"P{i+1}", col] = p_val
    return corr_results, k_results, p_results
