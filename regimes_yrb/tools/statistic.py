# !/usr/bin/env python
# - * - coding: utf-8 - * -
# author: Shuang Song time: 2020/1/14
# Email: SongshGeo@Gmail.com
# project: WGRegimes_YRB_2020

import itertools
import math
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


def line_fit(x, y):
    N = float(len(x))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for i in range(int(N)):
        sx += x[i]
        sy += y[i]
        sxx += x[i] * x[i]
        syy += y[i] * y[i]
        sxy += x[i] * y[i]
    a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
    # b = (sy - a * sx) / N
    r = abs(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
    return r"$k = %10.2f; R^2 =%10.2f$" % (a, r)


def get_optimal_fit_linear(x_arr, y_arr):
    from scipy import optimize

    def linear(x, slope, intercept):
        return slope * x + intercept

    k, b = optimize.curve_fit(linear, x_arr, y_arr)[0]  # optimize
    y_sim = linear(x_arr, k, b)  # simulated y
    return y_sim, k, b


def compile_cols(data, col1, col2):
    # 需要对比的值为value_x和value_y
    # 新家的列名为value_final
    # 1.设置一个flag,值为value_y-value_x,为正代表y较大，负代表x较大
    data["value_flag"] = data[col1] - data[col2]
    # 2.分别取得y较大的部分和x较大的部分
    df_test_bigger = data[data["value_flag"] >= 0].copy()
    df_test_litter = data[data["value_flag"] < 0].copy()
    # 3.分别对final进行赋值
    df_test_bigger["Value_Final"] = df_test_bigger[col1]
    df_test_litter["Value_Final"] = df_test_litter[col2]
    return pd.concat([df_test_bigger, df_test_litter])


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def huanbi_rate(data, use_col, new_col):
    data = data.reset_index()
    result = []
    for i in range(len(data)):
        if i == 0:
            rate = np.nan
        else:
            rate = format(
                (data[use_col][i] - data[use_col][i - 1]) / data[use_col][i - 1],
                ".2%",
            )
        result.append(rate)
    data[new_col] = result


def difference(data, use_col, new_col):
    data = data.reset_index()
    result = []
    for i in range(len(data)):
        if i == 0:
            rate = np.nan
        else:
            rate = format((data[use_col][i] - data[use_col][i - 1]), ".2%")
        result.append(rate)
    data[new_col] = result


# Pettitt 断点检测法
def pettitt_change_point_detection(inputdata, p_shr=0.05):
    """使用一次 Pettitt 断点检测"""
    index = inputdata.index
    inputdata = np.array(inputdata.values)
    n = inputdata.shape[0]
    k = range(n)
    inputdataT = pd.Series(inputdata)
    r = inputdataT.rank()
    Uk = [2 * np.sum(r[:x]) - x * (n + 1) for x in k]
    Uka = list(np.abs(Uk))
    U = np.max(Uka)
    K = index[Uka.index(U)]
    pvalue = 2 * np.exp((-6 * (U**2)) / (n**3 + n**2))
    change_point_desc = "显著" if pvalue <= p_shr else "不显著"
    pettitt_result = {"突变点位置": K, "突变程度": change_point_desc}
    return K, pettitt_result


# 循环检测所有显著的断点
def pettitt_changes(inputdata, p_shr=0.001):
    """根据给定阈值进行循环，找到所有断点"""
    change_points = []
    detect_series = [inputdata]
    while detect_series:
        tmp_detect = []
        for series in detect_series:
            k, pettitt_result = pettitt_change_point_detection(series, p_shr=p_shr)
            if pettitt_result["突变程度"] == "显著":
                change_points.append(k)
                tmp_detect.extend([series.loc[: k - 1], series.loc[k + 1 :]])
            else:
                continue
        detect_series = tmp_detect
        continue
    return change_points


def plot_pettitt_change_points(
    series,
    ax=None,
    p_shr=0.001,
    change_points=None,
    colors=None,
    returns="slopes",
    legend=True,
    **kargs,
):
    """将所有断点直观地展示出来"""
    from scipy import optimize

    def linear(x, k, b):
        return k * x + b

    slopes = []
    if change_points is None:
        change_points = sorted(pettitt_changes(series, p_shr))
    change_points_index = [series.index.tolist().index(i) for i in change_points]
    x_arr = np.split(series.index, change_points_index)
    y_arr = np.split(series.values, change_points_index)
    if ax is None:
        _, ax = plt.subplots()
    i = 0
    legends = []
    for xi, yi in zip(x_arr, y_arr):
        i += 1
        k, b = optimize.curve_fit(linear, xi, yi)[0]  # 最小二乘拟合直线
        y_simu = linear(xi, k, b)  # 拟合直线的预测Y
        slopes.append(k)

        # 绘图
        if colors:
            a = ax.scatter(
                xi,
                yi,
                label="P{}: {}-{}".format(i, xi[0], xi[-1]),
                color=colors[i - 1],
                **kargs,
            )  # 源数据散点图
            ax.plot(xi, y_simu, "--", color=colors[i - 1])  # 拟合直线图
        else:
            a = ax.scatter(xi, yi, label="P{}: {}-{}".format(i, xi[0], xi[-1]), **kargs)
            ax.plot(xi, y_simu, "--")
        legends.append(a)
    if legend:
        ax.legend()
    # y_position = ax.get_ylim()[1]
    for cp in change_points:
        ax.axvline(x=cp, ls=":", c="gray", lw=2)
        # ax.text(cp, y_position, "{}".format(cp), va='top', ha='center')  注释年份
    if returns == "slopes":
        return slopes
    elif returns == "legend":
        return legends


def plot_slopes(slopes, ax=None, colors=None):
    index = ["P{}".format(i + 1) for i in range(len(slopes))]
    slopes = pd.Series(slopes, index=index).sort_index(ascending=False)
    if ax is None:
        fig, ax = plt.subplots()
    if colors:
        color_list = colors.copy()
        color_list.reverse()
        slopes.plot.barh(color=color_list, ax=ax, edgecolor="lightgray")
    else:
        slopes.plot.barh(color="gray", ax=ax, edgecolor="lightgray")
    ax.axvline(x=0, c="gray", lw=2, ls="--")


# 计算分子除以分母各自的贡献
def ratio_contribution(numerator, denominator, breakpoints):
    """numerator 是一个 DataFrame，index 应该和分母一样，列可以有多个，是加总的关系"""
    if len(numerator.shape) == 1:  # 这时候传入的是一个 Series
        ratio = numerator / denominator
        numerator_sum = numerator.copy()
    else:  # 这时候传入的是一个 DataFrame
        numerator_sum = numerator.sum(axis=1)
        ratio = numerator_sum / denominator

    periods, changes = [], {}
    result = {}

    change_points = breakpoints.copy()
    change_points.extend([ratio.index.max(), ratio.index.min()])  # 为断点增加头尾
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

        # 每个阶段的正常单位变化量
        changes[period] = ratio.loc[end_year] - ratio.loc[start_year]
        result[period] = {}

        # 每个阶段的 ln 变化量
        ratio_change = np.log(ratio.loc[end_year]) - np.log(ratio.loc[start_year])
        # 分子分母的变化量
        numerator_change = np.log(
            numerator_sum.loc[end_year] / numerator_sum.loc[start_year]
        )
        denominator_change = np.log(
            denominator.loc[start_year] / denominator.loc[end_year]
        )

        # 分子分母各自的贡献率
        result[period]["Total"] = changes[period]
        result[period]["Numerator"] = changes[period] * numerator_change / ratio_change
        result[period]["Denominator"] = (
            changes[period] * denominator_change / ratio_change
        )

        # 各个分子的贡献率
        if len(numerator.shape) > 1:
            numerator_ratio = (
                numerator.loc[start_year:end_year, :].sum()
                / numerator.loc[start_year:end_year, :].sum().sum()
            )
            for col in numerator.columns:
                result[period][col] = (
                    numerator_ratio.loc[col] * result[period]["Numerator"]
                )

    return pd.DataFrame(result)


def plot_ratio_contribution(
    contribution_df,
    ax=None,
    bar_width=0.4,
    denominator_label="Denominator",
    colors=None,
    denominator_color=None,
    legend_loc=1,
):
    if colors is None:
        colors = ["#0077b6", "#e07a5f", "#f2cc8f", "#81b29a"]
    if denominator_color is None:
        denominator_color = "c"
    if ax is None:
        _, ax = plt.subplots()
    for i in range(len(contribution_df.columns)):
        numerator_position = i - bar_width / 2
        denominator_position = i + bar_width / 2

        period = contribution_df.columns[i]
        if len(contribution_df) == 3:  # 说明分子只有一个
            ax.bar(
                x=numerator_position,
                height=contribution_df.loc["Numerator", period],
                label=denominator_label,
                width=bar_width,
                color=colors[0],
                edgecolor="lightgray",
            )

        else:  # 分子有多个需要分别计算贡献
            bottom = 0
            color_index = 0
            for j in range(3, len(contribution_df)):
                height = contribution_df.iloc[j, i]
                # bar = ax.bar(x=numerator_position,
                #              height=height,
                #              bottom=bottom,
                #              label=contribution_df.index[j],
                #              width=bar_width,
                #              color=colors[color_index],
                #              edgecolor='lightgray')
                bottom += height
                color_index += 1
        ax.bar(
            x=denominator_position,
            height=contribution_df.loc["Denominator", period],
            width=bar_width,
            color=denominator_color,
            label=denominator_label,
            edgecolor="lightgray",
        )

        if i == 0:
            ax.legend(loc=legend_loc)
    ax.set_xticks(np.arange(0, len(contribution_df.columns), 1))
    ax.set_xticklabels(contribution_df.columns.tolist())


def judge_one_way_f_test(group1, group2):
    from scipy import stats

    F, p = stats.f_oneway(group1, group2)
    F_test = stats.f.ppf((1 - 0.05), 4, 15)
    print("F值是{:.2f}，p值是{:.5f}".format(F, p))
    print("F_test的值是{:.2f}".format(F_test))
    if F >= F_test:
        print("拒绝原假设，两组均值不相等, 显著性是{:.5f}".format(p))
        return False
    else:
        print("接受原假设，两组均值可认为是相等，显著性是{:.5f}".format(p))
        return True


def zscore(arr: np.ndarray) -> np.ndarray:
    """最大-最小值归一化，所有数据映射到0-1之间"""
    return (arr - arr.min()) / (arr.max() - arr.min())


def norm_zscore(arr: np.ndarray) -> np.ndarray:
    """先取对数，再最大-最小值归一化，所有数据映射到0-1之间"""
    return zscore(np.log(arr))


def calc_correlation(
    data: pd.DataFrame, columns: list = None, significance_level: float = 0.01
) -> pd.DataFrame:
    """
    计算 DataFrame 中指定的 4 列之间两两的相关系数和显著性水平。

    参数:
        data (pd.DataFrame): 输入的 DataFrame。
        columns (list): 列名列表，包含需要计算相关系数的 4 列。
        significance_level (float): 显著性水平（0 到 1 之间的值，例如 0.05）。

    返回:
        dict: 关于相关系数和显著性的字典。
    """

    if columns is None:
        columns = data.select_dtypes(include="number").columns.tolist()

    correlations = {}

    for col1, col2 in itertools.combinations(columns, 2):
        corr, p_value = pearsonr(data[col1], data[col2])

        is_significant = (
            "Significant" if p_value < significance_level else "Not Significant"
        )
        correlations[f"{col1} vs {col2}"] = {
            "correlation": corr,
            "p_value": p_value,
            "significance": is_significant,
        }

    return pd.DataFrame(correlations)


def calc_contribution_ratio(
    data: pd.DataFrame,
    columns: Optional[list] = None,
    start_yr: Optional[int] = None,
    end_yr: Optional[int] = None,
) -> pd.DataFrame:
    """ """
    if columns is None:
        columns = data.select_dtypes(include=["number"]).columns.to_list()
    if start_yr is None:
        start_yr = data.index.min()
    if end_yr is None:
        end_yr = data.index.max()

    # 检查输入数据是否包含 A、B 和 C 列
    if not all(col in data.columns for col in columns):
        raise ValueError("Please ensure the input DataFrame contains columns.")

    # 根据指定的时间范围筛选数据
    data = data.loc[start_yr:end_yr]

    # 计算 SUM 列的变化量
    delta_sum = data[columns].mean(axis=1).diff().sum()

    contribution_ratio = {}
    # 计算 A、B 和 C 三列的变化量
    for col in columns:
        delta = data[col].diff().sum() / len(columns)
        contribution_ratio[col] = delta / delta_sum
    return pd.Series(contribution_ratio)
