from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from functools import reduce
import numpy as np
import pandas as pd

# 使用sklearn中的iris数据集，通过numpy实现朴素贝叶斯

# 概率密度函数
def calculateProb(x, mean, std):
    exponent = np.exp(-(x-mean)**2/ (2 * std ))
    p = (1 / np.sqrt(2 * np.pi * std )) * exponent
    return p

def probability(x,mean,std):
    #  概率密度函数的连乘
    c=1
    for k in range(len(mean)):
        c*=calculateProb(x[k],mean[k],std[k])
    return c

def prior_probability(data, datas):
    # 计算先验概率
    return len(data)/len(datas)
# 数据处理
a=load_iris()
data=a.data
features_n=a.feature_names
target=a.target
train_data,test_data,train_target,test_target=train_test_split(data,target,train_size=0.5,test_size=None)


class beyes:
    def __init__(self):
        #在类中传递值
        self.tarin_data=None
        self.tarin_target=None
        self.mean=[]
        self.std=[]
        self.label=[]
    def tarin(self,train_data,train_target):
        self.label=set(sorted(list(train_target)))
        self.tarin_data=train_data
        self.tarin_target=train_target
        for kind in self.label:
            X=train_data[train_target==kind]
            mean=[pd.Series(X[:,last_feature]).mean() for last_feature in range(X.shape[-1])]
            std=[pd.Series(X[:,last_feature]).var() for last_feature in range(X.shape[-1])]
            self.std.append(std)
            self.mean.append(mean)

# mean是3*4
# test是n*4


    def predict(self,test_data,test_target):
        #  输入测试数据集，通过列表获得测试集对与不同类的预测结果，选取最大的
        prior=[]
        result=[]
        func=lambda x,y: x*y
        f=lambda x, y, z: calculateProb(x, y, z)
        self.mean=np.array(self.mean)
        self.std=np.array(self.mean)
        for l in self.label:
            answer = []
            for i in test_data:
                answer.append(reduce(func,map(f,i,self.mean[l],self.std[l])))
            prior.append(prior_probability(self.tarin_data[self.tarin_target == l], self.tarin_target))
            result.append(answer)
        result=np.array(result)
        for l in self.label:
            result[l]=result[l]*prior[l]


        # answer = pd.DataFrame(answer)
        answer = np.array(result)
        c = answer.argmax(axis=0)
        count = 0
        for pp in range(len(c)):
            if c[pp] == test_target[pp]: count += 1
            # print(c[c[pp]==2])
        print('准确率')
        print(count / len(c))


c=beyes()
c.tarin(train_data,train_target)
c.predict(test_data=test_data,test_target=test_target)


# predict()