import numpy as np
import random
import pyecharts.options as opts
from pyecharts.charts import Scatter
import pandas as pd
# 获取数据点（二维）
def getData(file):
    df=pd.read_csv(file,error_bad_lines=False)
    return np.array(df)

# 获取两点距离
def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

# 确定初始簇欣
def getOriginalCenter():
    k_center=[]
    '''
        初始化簇心
    '''
    return k_center

# K-means算法实现
def kMeans(dataset,k):
    # 这里要重写簇心初始化代码
    centerList = getOriginalCenter()
    center_change = True
    distance_List = np.full((1,3),-1)
    count = 0
    while center_change:
        count+=1
        print('第',count,'次训练')
        center_change = False
        # 更新簇
        for point in range(0,len(dataset)):
            for i in range(0,len(centerList)):
                distance_List[0,i] = distance(dataset[point],centerList[i])
            minIndex = np.argmin(distance_List)
            dataset[point][2] = minIndex

        # 重新计算簇心
        for i in range(0,k):
            xSum=0
            ySum=0
            psum=0
            for point in dataset:
                if point[2] == i:
                    xSum += point[0]
                    ySum += point[1]
                    psum += 1
            xAve = int(xSum / psum)
            yAve = int(ySum / psum)
            newCenter = np.array([xAve,yAve])
            if (newCenter != centerList[i]).all():
                center_change = True
            centerList[i] = newCenter
    return dataset

