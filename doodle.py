# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:50:35 2022

@author: taeni
"""

cfg = m4cfg
dataset = m4dataset
model_type='interpretable'

seasonal_pattern = cfg.seasonal_patterns[0]
lookback = cfg.lookbacks[0]
loss = cfg.losses[0]

from torchinfo import summary
summary(model, )


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\taeni\Documents\GitHub\N-BEATS\datasets\m4\m4_all.csv")

sp = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
h = [6, 8, 18, 13, 14, 48]
df_y = df[df['SP'] == sp[0]].dropna(axis=1)
df_q = df[df['SP'] == sp[1]].dropna(axis=1)
df_m = df[df['SP'] == sp[2]].dropna(axis=1)
df_w = df[df['SP'] == sp[3]].dropna(axis=1)
df_d = df[df['SP'] == sp[4]].dropna(axis=1)
df_h = df[df['SP'] == sp[5]].dropna(axis=1)

df_y['mean'] = df_y.iloc[:, 5:].sum(axis=1) / h[0]
df_q['mean'] = df_q.iloc[:, 5:].sum(axis=1) / h[1]
df_m['mean'] = df_m.iloc[:, 5:].sum(axis=1) / h[2]
df_w['mean'] = df_w.iloc[:, 5:].sum(axis=1) / h[3]
df_d['mean'] = df_d.iloc[:, 5:].sum(axis=1) / h[4]
df_h['mean'] = df_h.iloc[:, 5:].sum(axis=1) / h[5]

summary_y = df_y.iloc[:, 5:].describe()
summary_q = df_q.iloc[:, 5:].describe()
summary_m = df_m.iloc[:, 5:].describe()
summary_w = df_w.iloc[:, 5:].describe()
summary_d = df_d.iloc[:, 5:].describe()
summary_h = df_h.iloc[:, 5:].describe()


# distploting - seasonal
sns.distplot(df_y['mean'], color="blue", label=sp[0], hist=False)
sns.distplot(df_q['mean'], color="red", label=sp[1], hist=False)
sns.distplot(df_m['mean'], color="green", label=sp[2], hist=False)
sns.distplot(df_w['mean'], color="yellow", label=sp[3], hist=False)
sns.distplot(df_d['mean'], color="black", label=sp[4], hist=False)
sns.distplot(df_h['mean'], color="orange", label=sp[5], hist=False)
plt.title("Distribution")
plt.legend(title="seasonality")
plt.xlim(-50000, 50000)
plt.show()


## describe by category
dfg_y = df_y['mean'].groupby(df_y['category']).describe()
dfg_q = df_q['mean'].groupby(df_q['category']).describe()
dfg_m = df_m['mean'].groupby(df_m['category']).describe()
dfg_w = df_w['mean'].groupby(df_w['category']).describe()
dfg_d = df_d['mean'].groupby(df_d['category']).describe()
dfg_h = df_h['mean'].groupby(df_h['category']).describe()


###############################################################################
# outlier
test = df_y[df_y['mean']>50000]
test = df_y[df_y['mean']<100]


###############################################################################
# sample plot
import random
import pandas as pd
import matplotlib.pyplot as plt

df_all = pd.read_csv(r"C:\Users\taeni\Documents\GitHub\N-BEATS\datasets\m4\m4_all.csv")
train_len = [len(t) for t in m4dataset.trainset]
df_all['length'] = pd.DataFrame({'length':train_len})

sp = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
color = ['blue', 'red', 'green', 'violet', 'black', 'orange']
h = [6, 8, 18, 13, 14, 48]
category = list(set(m4dataset.info.category.values))
category.sort()

trainset = m4dataset.trainset
testset = m4dataset.testset
for i, s in enumerate(sp):
    plt.figure(figsize=(50,30))
    for j, c in enumerate(category):
        if s=='Hourly' and c!='Other':
            continue
        df = df_all[df_all['SP'] == s]
        df = df[df['category']==c]
        idx = list(df.index)
        lst_sample=random.sample(range(0, len(df)), 5)
        lst_sample.sort()
        for k, l in enumerate(lst_sample):
            plt.subplot(6,5,(j*5)+k+1)
            tt = np.append(trainset[idx[l]], testset[idx[l]])
            ddf = pd.DataFrame(tt, columns=['data'])    
            plt.plot(ddf['data'][:-h[i]], color='blue', label='train')
            plt.plot(ddf['data'][-h[i]:], color='red', label='horizon')
            plt.legend()
            plt.title(c+'-'+str(l+1))                      
    plt.show()


###############################################################################
# length plot - seasonality
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_all = pd.read_csv(r"C:\Users\taeni\Documents\GitHub\N-BEATS\datasets\m4\m4_all.csv")
train_len = [len(t) for t in m4dataset.trainset]
df_all['length'] = pd.DataFrame({'length':train_len})

sp = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
color = ['blue', 'red', 'green', 'violet', 'black', 'orange']
category = list(set(m4dataset.info.category.values))
category.sort()

plt.figure(figsize=(7, 30))
for i, s in enumerate(sp):
    df = df_all[df_all['SP'] == s]
    plt.subplot(6,1,i+1)
    sns.kdeplot(df['length'], color=color[i], label=s)
    #plt.title(s + ' length distribution')
    plt.legend()
plt.show()

# length plot - seasonality & category
plt.figure(figsize=(50, 30))
for i, s in enumerate(sp):
    for j, c in enumerate(category):
        df = df_all[df_all['SP'] == s]
        df = df[df['category']==c]

        n = i*6+j+1
        if n==31 or n==32 or n==33 or n==34 or n==35:
            continue        
        plt.subplot(6,6,n)
        sns.kdeplot(df['length'], color=color[i], label=s+'-'+c)
        #plt.title(s + ' length distribution')
        plt.legend()
plt.show()
    
###############################################################################
# Value plot - seasonality
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_all = pd.read_csv(r"C:\Users\taeni\Documents\GitHub\N-BEATS\datasets\m4\m4_all.csv")
train_len = [len(t) for t in m4dataset.trainset]
df_all['mean'] = df_all.iloc[:, 5:].mean(axis=1)
df_all['length'] = pd.DataFrame({'length':train_len})

sp = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
color = ['blue', 'red', 'green', 'violet', 'black', 'orange']
category = list(set(m4dataset.info.category.values))
category.sort()

plt.figure(figsize=(7, 30))
for i, s in enumerate(sp):
    df = df_all[df_all['SP'] == s]
    plt.subplot(6,1,i+1)
    sns.kdeplot(df['mean'], color=color[i], label=s)
    #plt.title(s + ' length distribution')
    plt.legend()
plt.show()

# mean plot - seasonality & category
plt.figure(figsize=(50, 30))
for i, s in enumerate(sp):
    for j, c in enumerate(category):
        df = df_all[df_all['SP'] == s]
        df = df[df['category']==c]

        n = i*6+j+1
        if n==31 or n==32 or n==33 or n==34 or n==35:
            continue        
        plt.subplot(6,6,n)
        sns.kdeplot(df['mean'], color=color[i], label=s+'-'+c)
        #plt.title(s + ' length distribution')
        plt.legend()
plt.show()



###############################################################################
## Matrix-Matrix Multiplication
import numpy as np

A = np.array([[1,2,3], [4,5,6]])
B = A.transpose()
R = np.einsum('ik,kj->ij', A, B)


## Trendbasis  model init
import torch as t
import numpy as np

degree_of_polynomial = 2
backcast_size = 12
forecast_size = 6

polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
backcast_time = t.nn.Parameter(
    t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                             for i in range(polynomial_size)]), dtype=t.float32),
    requires_grad=False)
forecast_time = t.nn.Parameter(
    t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                             for i in range(polynomial_size)]), dtype=t.float32), requires_grad=False)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(forecast_time.transpose(0,1))
plt.plot(backcast_time.transpose(0,1))




###############################################################################
## Sesonally model init
import torch as t
import numpy as np

harmonics=1
forecast_size=6
backcast_size=12

frequency = np.append(np.zeros(1, dtype=np.float32),
                           np.arange(harmonics, harmonics / 2 * forecast_size,
                                     dtype=np.float32) / harmonics)[None, :]
backcast_grid = -2 * np.pi * (
        np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
forecast_grid = 2 * np.pi * (
        np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * frequency

backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32), requires_grad=False)
backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32), requires_grad=False)
forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32), requires_grad=False)
forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32), requires_grad=False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(backcast_cos_template.transpose(0,1))
plt.plot(backcast_sin_template.transpose(0,1))
plt.plot(forecast_cos_template.transpose(0,1))
plt.plot(forecast_sin_template.transpose(0,1))



###############################################################################
## Sesonally model init

import tensorboard



