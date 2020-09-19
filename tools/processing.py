# !/usr/bin/env python
# - * - coding: utf-8 - * -
# author: Shuang Song time: 2020/1/14
# Email: SongshGeo@Gmail.com
# project: 

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv

# 所有黄河流域的省份
PROVINCES = [
    '青海', '青海省', 'Qinghai',
    '四川', '四川省', 'Sichuan',
    '甘肃', '甘肃省', 'Gansu',
    '宁夏', '宁夏回族自治区', 'Ningxia',
    '内蒙', '内蒙古', '内蒙古自治区', 'Neimeng',
    '陕西', '陕西省', 'Shanxi', 
    '山西', '山西省', 'Shaanxi',
    '河南', '河南省', 'Henan',
    '山东', '山东省', 'Shandong'
    '河北', '河北省', 'Hebei'
]


period_colors = ['#0889A6', '#F1801F', '#006C43']
region_colors = ["#0077b6", "#e07a5f", "#f2cc8f","#81b29a"]
index_colors = ['#CFA7D1', '#79D6F0', '#E25A53']
import matplotlib
index_colormap = matplotlib.colors.ListedColormap(index_colors, 'indexed')
total_water_use_color = '#D1495B'


# 划分子图
def add_subplot_by_order(gs=(2, 2), order=1):
    a, b = gs
    x = (order - 1) // b  # 第几排
    y = order % b - 1
    return x, y


# 根据省份判断所在区域
def get_region_by_province_name(name):
    """使用省份名称获取该省属于哪个区域（SR, UR, MR or DR）
    name: 一个中国黄河流域的省份
    return: 区域（源区上中下游，或返回空值）
    """
    if name in PROVINCES[:6]:
        return 'SR'
    elif name in PROVINCES[6:16]:
        return 'UR'
    elif name in PROVINCES[16:22]:
        return 'MR'
    elif name in PROVINCES[22:]:
        return 'DR'
    else:
        return None


# 将dbf文件读取入pandas.DataFrame
def pd_read_dbf(io, usecols=False):
    """从文件系统中读取dbf文件并写入DataFrame
    io: 文件路径，
    usecols: 使用的列"""
    from dbfread import DBF  # 加载dbfreead包才能读取文件
    if io.endswith('.dbf'):
        table = DBF(io)
    else:
        raise Exception("{} is not a dbf file.".format(io.split("/")[-1]))
    df = pd.DataFrame(table)
    if usecols:
        for col in df:
            if col not in usecols:
                df.drop(col, axis=1, inplace=True)
    return df


def dbf_data_list(folder_path):
    """使用保存dbf文件的路径返回所有dbf文件列表
    folder_path: 装有dbf文件的文件夹"""
    dbf_files = []
    full_list = os.listdir(folder_path)
    for file in full_list:
        if file.endswith(".dbf"):
            dbf_files.append(file)
    return dbf_files


# 使用图片的比例来定位
def get_position_by_ratio(ax, x_ratio, y_ratio):
     x_min, x_max = ax.get_xlim()
     y_min, y_max = ax.get_ylim()
     x = (x_max-x_min) * x_ratio + x_min
     y = (y_max-y_min) * y_ratio + y_min
     return x, y