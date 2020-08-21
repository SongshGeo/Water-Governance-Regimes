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


def get_position_by_ratio(ax, x_ratio, y_ratio):
     x_min, x_max = ax.get_xlim()
     y_min, y_max = ax.get_ylim()
     x = (x_max-x_min) * x_ratio + x_min
     y = (y_max-y_min) * y_ratio + y_min
     return x, y