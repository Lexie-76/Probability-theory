import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#读取十年期国债周收益率数据
data = pd.read_csv("data.csv", encoding="ISO-8859-1")
#直方图
import seaborn as sns
sns.displot(data['r'], color="b", bins=10, kde=True)
plt.show()
#是否正态
from scipy import stats
res1 = stats.kstest(data['r'], 'norm')
print(res1)
res2 = stats.shapiro(data['r'])
print(res2)
#正态拟和
plt.figure()
data['r'].plot(kind = 'kde')
M_S = stats.norm.fit(data['r'])
normal_distribution = stats.norm(M_S[0], M_S[1])
x = np.linspace(normal_distribution.ppf(0.01), normal_distribution.ppf(0.99), 100)
plt.plot(x, normal_distribution.pdf(x), c='orange')
plt.xlabel('r')
plt.title('r on normal_distribution', size=20)
plt.legend(['Origin', 'normal_distribution'])
plt.show()
#是否t分布
np.random.seed(1)
ks = stats.t.fit(data['r'])
df = ks[0]
loc = ks[1]
scale = ks[2]
t_estm = stats.t.rvs(df=df, loc=loc, scale=scale, size=len(data['r']))
res3 = stats.ks_2samp(data['r'], t_estm)
print(res3)
#t分布拟和
plt.figure()
data['r'].plot(kind = 'kde')
t_distribution = stats.t(ks[0], ks[1],ks[2])
x = np.linspace(t_distribution.ppf(0.01), t_distribution.ppf(0.99), 100)
plt.plot(x, t_distribution.pdf(x), c='orange')
plt.xlabel('r')
plt.title('r on t_distribution', size=20)
plt.legend(['data', 't_distribution'])
plt.show()

