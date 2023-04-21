#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import os
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import ternary  # pip install python-ternary
from hydra import compose, initialize
from matplotlib.gridspec import GridSpec
from scipy import stats

from regimes_yrb.tools.statistic import (
    get_optimal_fit_linear,
    pettitt_changes,
    plot_pettitt_change_points,
    plot_ratio_contribution,
    zscore,
)

# 加载项目层面的配置
with initialize(version_base=None, config_path="../config"):
    cfg = compose(config_name="config")
os.chdir(cfg.root)

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "out"

COLORS = cfg.style.colors
period_colors = COLORS.period
region_colors = COLORS.region
index_colors = COLORS.index

index_colormap = matplotlib.colors.ListedColormap(index_colors, "indexed")
total_water_use_color = COLORS.total_WU


def plot_data(
    index,
    contribution,
    ylabel1: str = "Index_Y_label",
    ylabel2: str = "Contribution_Y_label",
    denominator_label: str = "Denominator",
    change_points: Optional[List[int]] = None,
):
    """绘制指标的以及它们的贡献"""
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    plot_pettitt_change_points(
        index, change_points=change_points, ax=ax1, colors=period_colors
    )
    plot_ratio_contribution(
        contribution,
        ax=ax2,
        colors=region_colors,
        denominator_color=total_water_use_color,
        denominator_label=denominator_label,
    )

    ax1.set_ylabel(ylabel1)
    ax1.set_xlabel("Year")

    ax2.set_xlabel("Different periods")
    ax2.set_ylabel(ylabel2)
    ax2.tick_params(axis="x", tickdir="in", bottom=False, labelrotation=0)
    ax2.axhline(0.0, lw=2, color="gray")
    # ax2.set_ylim(-0.08, 0.08)
    # ax2.set_yticks(np.arange(-0.08, 0.081, 0.04))
    ax2.axvline(1.5, ls=":", color="gray", lw=1.5)
    ax2.axvline(0.5, ls=":", color="gray", lw=1.5)

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(True)
    return ax1, ax2


# 制作三元图
def plot_ternary(data, ax=None):
    """绘制综合指数的三元图"""
    priority = data["P"]
    allocation = data["A"]
    scarcity = data["S"]
    # 开始、结束的时间
    start, end = min(data.index), max(data.index)
    breakpoints = pettitt_changes(data["IWGI"])
    if not ax:
        _, ax = plt.subplots(figsize=(4, 4))
    _, tax = ternary.figure(ax=ax, scale=1)
    annotates = [start, *breakpoints, end]

    points = []
    points_1, points_2, points_3 = [], [], []
    size_1, size_2, size_3 = [], [], []
    scale = 50
    for yr in data.index:
        sumed = priority[yr] + allocation[yr] + scarcity[yr]  # 这里点的大小是三者 norm_score 相加
        point = (
            priority[yr] / sumed,
            allocation[yr] / sumed,
            scarcity[yr] / sumed,
        )
        if yr in annotates:
            tax.annotate(text=yr, position=point)
        points.append(point)
        if yr < breakpoints[0]:
            points_1.append(point)
            size_1.append(sumed * scale)
        elif yr < breakpoints[1]:
            points_2.append(point)
            size_2.append(sumed * scale)
        else:
            points_3.append(point)
            size_3.append(sumed * scale)

    tax.boundary()
    tax.gridlines(ls="-.", multiple=1.0 / 3, color="black")
    # Plot a few different styles with a legend
    # tax.scatter(points_1, marker='o', color=period_colors[0], label="P1", s=scale, alpha=.4)
    # tax.scatter(points_2, marker='o', color=period_colors[1], label="P2", s=scale, alpha=.4)
    # tax.scatter(points_3, marker='o', color=period_colors[2], label="P3", s=scale, alpha=.4)
    tax.plot(
        points_1,
        ls="--",
        marker="o",
        color=period_colors[0],
        lw="1.5",
        label="P1",
        alpha=0.6,
    )  # 各点之间的连接线
    tax.plot(
        points_2,
        ls="--",
        marker="o",
        color=period_colors[1],
        lw="1.5",
        label="P2",
        alpha=0.6,
    )  # 各点之间的连接线
    tax.plot(
        points_3,
        ls="--",
        marker="o",
        color=period_colors[2],
        lw="1.5",
        label="P3",
        alpha=0.6,
    )  # 各点之间的连接线

    fontsize = 10
    offset = 0.15

    tax.ticks(
        axis="brl",
        multiple=1.0 / 3,
        linewidth=1,
        tick_formats="%.1f",
        offset=0.03,
    )
    # tax.ticks(axis='l', clockwise=True, multiple=1./3, linewidth=1, tick_formats="%.1f", offset=0.03)
    tax.get_axes().axis("off")
    tax.clear_matplotlib_ticks()
    tax.set_axis_limits((0, 1))
    #     tax.line((0,0,1),(.5,.5,0), ls=":", color='red', label='A=P')
    # tax.line((1,0,0),(0,.5,0.5), ls=":", color='black', label='A=S')
    # tax.annotate('Test', (0.3, 0.1, 0.6))
    # P A S

    tax.left_axis_label("Stress", fontsize=fontsize, offset=offset)
    tax.right_axis_label("Allocation", fontsize=fontsize, offset=offset)
    tax.bottom_axis_label("Purpose", fontsize=fontsize, offset=0.1)

    #     tax.right_corner_label("P", fontsize=fontsize, offset=0.25, weight='bold')
    #     tax.top_corner_label("A", fontsize=fontsize, offset=0.25, weight='bold')
    #     tax.left_corner_label("S", fontsize=fontsize, offset=0.25, weight='bold')
    #     ax.arrow(0.1, 0.3, 0.3, 0.4)
    tax.legend(loc=2)
    return tax


def plot_demonstrate(ax=None):
    """绘制示意图"""
    if not ax:
        _, ax = plt.subplots(figsize=(2, 2))
    _, tax = ternary.figure(ax=ax, scale=1)
    tax.boundary()
    tax.gridlines(ls="-.", multiple=1.0 / 3, color="gray")
    alpha = 0.8
    ax.arrow(
        0.5,
        0.01,
        0.22,
        0.30,
        lw=3,
        head_width=0.05,
        head_length=0.1,
        fc=period_colors[0],
        ec=period_colors[0],
        alpha=alpha,
    )
    ax.arrow(
        0.82,
        0.31,
        -0.45,
        -0.25,
        lw=3,
        head_width=0.05,
        head_length=0.1,
        fc=period_colors[1],
        ec=period_colors[1],
        alpha=alpha,
    )
    ax.arrow(
        0.27,
        0.03,
        -0.05,
        0.22,
        lw=3,
        head_width=0.05,
        head_length=0.1,
        fc=period_colors[2],
        ec=period_colors[2],
        alpha=alpha,
    )
    ax.text(0.55, 0.33, "P1", weight="bold", color=period_colors[0])
    ax.text(0.37, 0.17, "P2", weight="bold", color=period_colors[1])
    ax.text(0.25, 0.30, "P3", weight="bold", color=period_colors[2])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


def plot_iwgi_scatter(iwgi):
    """绘制散点图"""
    k_results = pd.DataFrame(index=["P1", "P2", "P3"])
    # b_results = pd.DataFrame(index=["P1", "P2", "P3"])
    corr_results = pd.DataFrame(index=["P1", "P2", "P3"])
    p_results = pd.DataFrame(index=["P1", "P2", "P3"])
    breakpoints = [1965, 1978, 2001, 2013]

    _, axs = plt.subplots(1, 3, figsize=(10, 3))
    for i, yr_start in enumerate(breakpoints[:3]):
        ax = axs[i]
        yr_end = breakpoints[i + 1]
        temp_data = iwgi.loc[yr_start:yr_end]
        for col in iwgi:
            x, y = temp_data.index, temp_data[col].values
            y_sim, k, _ = get_optimal_fit_linear(x, y)
            k_results.loc[f"P{i+1}", col] = k
            ax.scatter(zscore(temp_data[col]), zscore(y_sim), label=f"{col}")
            if col != "IWGI":
                corr, p_val = stats.spearmanr(
                    temp_data[col].values, temp_data["IWGI"].values
                )
                corr_results.loc[f"P{i+1}", col] = corr
                p_results.loc[f"P{i+1}", col] = p_val
        ax.legend()
