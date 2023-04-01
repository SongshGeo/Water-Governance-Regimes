#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Iterable, LiteralString, TypeAlias

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

Columns: TypeAlias = Iterable[LiteralString]

SITE_COLS = {
    "唐乃亥": "SR",  # 唐乃亥控制源区
    "头道拐": "UR",  # 头道拐控制上游
    "花园口": "MR",  # 花园口控制中游
    "利津": "DR",  # 利津控制下游
}

regions = ["SR", "UR", "MR", "DR"]


def measure_runoff(runoff, cite_cols):
    """加载径流数据"""
    # 加载径流和水库数据
    measured_runoff = pd.read_csv(runoff, index_col="年份")
    measured_runoff = measured_runoff.loc[:, cite_cols.keys()]
    measured_runoff.rename(cite_cols, axis=1, inplace=True)
    return measured_runoff


def load_reservoirs_data(reservoirs):
    """加载水库数据"""
    # 水库库容数据
    reservoirs_capacity = pd.read_csv(reservoirs, index_col=0)

    # 水库库容累积相加数据
    return reservoirs_capacity.cumsum()


def calculate_index_a(
    runoff: pd.DataFrame,
    consumptions: pd.DataFrame,
    total_col: str = "Total water use",
) -> pd.DataFrame:
    """
    计算指数 A: 总耗水量 / 平均多年径流量

    Args:
        measured_runoff (pd.DataFrame): 指示流域径流量的 DataFrame。
        consumptions (pd.DataFrame): 包含用水数据的 DataFrame。

    Returns:
        pd.DataFrame: 包含指数 A 的 DataFrame。
    """
    total_water_use = consumptions.pivot_table(
        index="Year", columns="Region", aggfunc=np.sum, values=total_col
    )
    results = total_water_use.apply(
        lambda row: row / runoff.mean(axis=0), axis=1
    )
    return results[["SR", "UR", "MR", "DR"]]


def calculate_index_b(runoff, consumptions, inflexible_wu) -> pd.DataFrame:
    """计算指数 B: 总不灵活耗水量 / 平均多年径流量"""
    consumptions["_inflexible_wu"] = consumptions.loc[:, inflexible_wu].sum(
        axis=1
    )
    inflexible_water_use = consumptions.pivot_table(
        index="Year",
        columns="Region",
        aggfunc=np.sum,
        values="_inflexible_wu",
    )
    del consumptions["_inflexible_wu"]
    results = inflexible_water_use.apply(
        lambda row: row / runoff.mean(axis=0), axis=1
    )
    return results[["SR", "UR", "MR", "DR"]]


def calculate_index_c(
    runoff: pd.DataFrame, reservoirs: pd.DataFrame
) -> pd.DataFrame:
    """
    计算指数 C。

    Args:
        measured_runoff (pd.DataFrame): 指示流域径流量的 DataFrame。
        reservoirs_capacity_cumulating (pd.DataFrame): 指示水库容量的 DataFrame。

    Returns:
        pd.DataFrame: 包含指数 C 的 DataFrame。
    """
    index_c1 = runoff.std() / runoff.mean()
    avg_runoff = runoff.mean(axis=0)
    index_c2 = reservoirs.apply(lambda row: row / avg_runoff, axis=1).clip(
        upper=1
    )
    return (1 - index_c2) * index_c1


def normalization_horizontal(data):
    """横向在四个区域之间进行标准化"""
    sfv = pd.DataFrame(columns=[regions])
    for row in data.itertuples():
        year = row[0]
        value = np.array(row[1:])
        avg = np.mean(value)
        std = np.std(value)
        sfv.loc[year] = (value - avg) / std
    return sfv


def calculate_sfv_indices(index_a, index_b, index_c):
    """计算sfv指数"""
    use_years = index_a.index.unique()
    v_values = np.zeros(index_a.shape)

    for index in [index_a, index_b, index_c]:
        index = normalization_horizontal(index.loc[use_years].copy())
        v_values += index.values / 3

    sfv_values = []
    for v_row in v_values:
        a = 1 / (v_row.max() - v_row.min())
        b = 1 * v_row.min() / (v_row.min() - v_row.max())
        sfv = a * v_row + b
        sfv_values.append(sfv)
    sfv_regions = pd.DataFrame(sfv_values, columns=regions, index=use_years)
    sfv = pd.Series(np.array(sfv_values).mean(axis=1), index=use_years)

    return sfv_regions, sfv


def get_index_yr(start_yr, end_yr, indexes):
    """从三个指标中提取指定时间范围的数据并标准化"""
    differences = {}
    for index, value in indexes.items():
        data = normalization_horizontal(value)
        differences[index] = (data.loc[end_yr] - data.loc[start_yr]) / 3
    return pd.Series(differences)


def calc_sfv(runoff, consumptions, reservoirs, inflexible_wu):
    """计算 SFV 指数，以及各指标的贡献"""
    index_a = calculate_index_a(runoff, consumptions)
    index_b = calculate_index_b(runoff, consumptions, inflexible_wu)
    index_c = calculate_index_c(runoff, reservoirs)
    indexes = {"A": index_a, "B": index_b, "C": index_c}
    _, sfv = calculate_sfv_indices(index_a, index_b, index_c)
    # TODO 这里算的贡献是固定时间段，不好
    sfv_contribution = pd.DataFrame()
    for breakpoints in [(1965, 1977), (1978, 1993), (1994, 2013)]:
        data = get_index_yr(breakpoints[0], breakpoints[1], indexes)
        sfv_contribution[f"{breakpoints[0]}-{breakpoints[1]}"] = data
    return sfv, sfv_contribution


# ================================== 作图 =============================
def sfv_contribution_plot(sfv_contribution, ax=None):
    """分解 SFV 中每个指标的贡献"""
    if not ax:
        _, ax = plt.subplots()
    width = 0.16
    alpha = 1
    for i in range(3):
        data = sfv_contribution.iloc[i * 3 : i * 3 + 3]  # 数据
        # period = "P{}".format(i + 1)  # 哪个阶段
        bottoms_pos = [0, 0, 0, 0]  # 正方向的底
        bottoms_neg = [0, 0, 0, 0]  # 反方向的底
        for j, row in enumerate(data.itertuples(index=False)):
            index = ["Scarcity", "Flexibility", "Variability"][j]
            color = ["#CFA7D1", "#79D6F0", "#E25A53"][j]
            for d, height in enumerate(row):
                region = ["SR", "UR", "MR", "DR"][d]
                position = i - 1.5 * width + d * width
                if j == 0 and i == 0:
                    ax.text(
                        position, 0.4, region, size=6, va="center", ha="center"
                    )
                if height >= 0:
                    bottom = bottoms_pos[d]
                    if d == 1 and i == 1:
                        ax.bar(
                            x=position,
                            height=height,
                            bottom=bottom,
                            width=width,
                            color=color,
                            alpha=alpha,
                            edgecolor="lightgray",
                            label=index,
                        )
                    else:
                        ax.bar(
                            x=position,
                            height=height,
                            bottom=bottom,
                            width=width,
                            color=color,
                            alpha=alpha,
                            edgecolor="lightgray",
                        )
                    bottoms_pos[d] += height
                else:
                    bottom = bottoms_neg[d]
                    if d == 1 and i == 1:
                        ax.bar(
                            x=position,
                            height=height,
                            bottom=bottom,
                            width=width,
                            color=color,
                            alpha=alpha,
                            edgecolor="lightgray",
                            label=index,
                        )
                    else:
                        ax.bar(
                            x=position,
                            height=height,
                            bottom=bottom,
                            width=width,
                            color=color,
                            alpha=alpha,
                            edgecolor="lightgray",
                        )
                    bottoms_neg[d] += height
    ax.legend()
    ax.axhline(y=0, ls="--", lw=2, color="gray")
    ax.axvline(1.5, ls=":", color="gray")
    ax.axvline(0.5, ls=":", color="gray")
    ax.set_ylabel("Index values")
    ax.set_xticks(np.arange(0, 2.1, 1))
    ax.set_xlabel("Periods")
    ax.set_xticklabels(["P1: 1965-1977", "P2: 1978-1993", "P3: 1994-2013"])

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(False)

    ax.text(0, 0.5, "Regions", va="bottom", ha="center", size=8)
    ax.annotate(
        "",
        xy=(-0.4, 0.45),
        xycoords="data",
        xytext=(0.4, 0.45),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"),
    )
    return ax
