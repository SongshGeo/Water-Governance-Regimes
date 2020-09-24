#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

import numpy as np
import pandas as pd


# 清洗列名
def from_col_name_extract_province_and_industry(name):
    elements = name.split(':')  # 分割列名
    province = elements[0].strip()  # 地名
    industry = elements[-1].strip()  # 产业
    return province, industry



# 根据年份，提取数据
def extract_data_by_yr(data, start_yr, end_yr):
    # 当需要提取的年份在起点年份之前，采用时间序列中最早的一年
    if start_yr not in data.index:
        start_yr = data.index[0]
    # 同理，如果终点年份超出索引，采用最晚的一年
    if end_yr not in data.index:
        end_yr = data.index[-1]

    # 当两者相等时，无标准差
    if start_yr == end_yr:
        return data.loc[end_yr], False
    else:
        use_data = data.loc[start_yr: end_yr]
        return np.mean(use_data), np.std(use_data)



def plot_bar_by_category(ax, data, categories, colors, position, start_yr, end_yr, **kargs):
    """ 绘制一个分类柱状图，将不同类别画在同一个位置上
    ax: 绘图区域
    data: 数据，DataFrame，index 是时间（年份），列与 categories 相关
    categories: 分类列表，其中的类应该是 data 中出现的列名
    colors: 每个类使用的颜色
    position: 这个分类柱状图的柱子位置
    start_yr: 截取数据的起始年
    end_yr: 截取数据的末位年
    **kargs: 与绘图相关的参数
    ---------------------------------------------
    return: 返回图例
    """
    bottom = 0  # 初始的柱状图底
    legends = []  # 图例
    for i, category in enumerate(categories):  # 对每个分类
        use_data = data[category]  # 数据是其中的类
        height, std = extract_data_by_yr(use_data, start_yr, end_yr)
        if std:  # 如果有标准差，就绘制误差线
            bar = ax.bar(
                x=position,
                bottom=bottom,
                height=height, 
                yerr=std,
                color=colors[i],
                **kargs
            )
        else:
            bar = ax.bar(
                x=position,
                bottom=bottom,
                height=height, 
                color=colors[i],
                **kargs
            )
        # 每次循环之后：
        bottom += height  # 原先的高作为新的底
        legends.append(bar)  # 图像存为图例
    return legends


# 根据指定的产业提取数据，计算区域总的 GDP
def extract_gdp_by_industry(data, industry):
    """ 根据指定的产业，提取GDP数据
    data: GDP 数据，列名格式为："地区: GDP: 第X产业"
    industry: 需要提取的产业
    --------------------------------
    return: 给定数据集中，所有该产业的逐年 GDP
    """
    df = pd.DataFrame()
    for col in data:
        if industry in col:
            df[col] = data[col]
        else:
            continue
    return df.sum(axis=1)


if __name__ == "__main__":
    pass