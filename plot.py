import matplotlib.pyplot as plt
import numpy as np

# # epoch,acc,loss,val_acc,val_loss
# x_axis_data = ['1st Round', '2nd Round', '3rd Round', '4th Round', '5th Round', '6th Round']
# y_axis_data1 = [45.51, 48.23, 50.55, 51.46, 51.97, 52.39]
# y_axis_data2 = [46.58, 48.96, 50.99, 52.58, 53.06, 53.68]
# y_axis_data3 = [48.74, 50.21, 51.73, 52.41, 52.87, 53.38]
# complete = [49.01, 51.69, 52.37, 53.18, 53.74, 54.17]

# # 画图
# plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=1, label='Baseline')  # '
# plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=1, label='w/ RWFSI')
# plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='w/ ICDSS')
# plt.plot(x_axis_data, complete, '^--', alpha=0.5, linewidth=1, label='Complete')

# ## 设置数据标签位置及大小
# for a, b in zip(x_axis_data, y_axis_data1):
#     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b1 in zip(x_axis_data, y_axis_data2):
#     plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
# for a, b2 in zip(x_axis_data, y_axis_data3):
#     plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
# for a, b3 in zip(x_axis_data, complete):
#     plt.text(a, b3, str(b3), ha='center', va='bottom', fontsize=8)
# plt.legend()  # 显示上面的label

# plt.ylabel('mIoU')  # accuracy

# # plt.ylim(-1,1)#仅设置y轴坐标范围
# plt.show()

#######################################################################################################################################################

# epoch,acc,loss,val_acc,val_loss
x_axis_data = ['[0.0,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)', '[0.4,0.5)', '[0.5,0.6)', '[0.6,0.7)', '[0.7,0.8)', '[0.8,0.9)', '[0.9,1.0]']
y_axis_data1 = [0.0287,0.0000,0.4833,0.5296,0.5621,0.6282,0.6827,0.7377,0.7707,0.7972]
y_axis_data2 = [0.2878,0.2060,0.4289,0.5153,0.5768,0.6186,0.6745,0.7293,0.7792,0.8103]
# y_axis_data3 = [48.74, 50.21, 51.73, 52.41, 52.87, 53.38]
# complete = [49.01, 51.69, 52.37, 53.18, 53.74, 54.17]

# 画图
plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=1, label='RIPU')  # '
plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=1, label='BADA (Ours)')
# plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='w/ ICDSS')
# plt.plot(x_axis_data, complete, '^--', alpha=0.5, linewidth=1, label='Complete')

## 设置数据标签位置及大小
for i in range(len(y_axis_data1)):
    a, b1, b2 = x_axis_data[i], y_axis_data1[i], y_axis_data2[i]
    if b1>b2:
        plt.text(a, b1+0.01, str(b1), ha='center', va='bottom', fontsize=8, color='b', alpha=0.5)
        plt.text(a, b2-0.01, str(b2), ha='center', va='top', fontsize=8, color='r', alpha=0.5)
    else:
        plt.text(a, b1-0.01, str(b1), ha='center', va='top', fontsize=8, color='b', alpha=0.5)
        plt.text(a, b2+0.01, str(b2), ha='center', va='bottom', fontsize=8, color='r', alpha=0.5)

# for a, b in zip(x_axis_data, y_axis_data1):
#     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b1 in zip(x_axis_data, y_axis_data2):
#     plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
# for a, b2 in zip(x_axis_data, y_axis_data3):
#     plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
# for a, b3 in zip(x_axis_data, complete):
#     plt.text(a, b3, str(b3), ha='center', va='bottom', fontsize=8)
plt.legend()  # 显示上面的label
plt.xticks(rotation=35)

plt.xlabel('Softmax entropy')
plt.ylabel('Proportion of selected misclassified samples')

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()

#######################################################################################################################################################

# # 创建数据
# rng = np.random.RandomState(10)
# data1 = rng.normal(size=1000)
# data2 = rng.normal(size=1000)
# print(data1.shape)

# # data1 = np.array([0.33476305, 0.06301446, 0.21919415, 0.01559775, 0.01121762, 0.01073472,
# #  0.0032475,  0.00494992, 0.10995878, 0.00698865, 0.14604204, 0.00216487,
# #  0.0003897,  0.05956245, 0.00244388, 0.00256802, 0.00519101, 0.00035185,
# #  0.00161958])
# # data2 = np.array([0.19893065, 0.12219848, 0.19141214, 0.03757275, 0.03137845, 0.04423565,
# #  0.00857152, 0.01437508, 0.14521885, 0.01522797, 0.10153866, 0.00825205,
# #  0.00168479, 0.06111166, 0.00276793, 0.00315923, 0.00696321, 0.00166403,
# #  0.0037369])

# # 创建分割
# # binRange = np.arange(-4,4,1)
# # hist1,_ = np.histogram(data1, bins=binRange)
# # hist2,_ = np.histogram(data2, bins=binRange)
# # ratio = hist2/(hist1+hist2) # 所占比例

# binRange = np.arange(0,20,1)
# hist1 = np.array([0.33476305, 0.06301446, 0.21919415, 0.01559775, 0.01121762, 0.01073472,
#  0.0032475,  0.00494992, 0.10995878, 0.00698865, 0.14604204, 0.00216487,
#  0.0003897,  0.05956245, 0.00244388, 0.00256802, 0.00519101, 0.00035185,
#  0.00161958])
# hist2 = np.array([0.19893065, 0.12219848, 0.19141214, 0.03757275, 0.03137845, 0.04423565,
#  0.00857152, 0.01437508, 0.14521885, 0.01522797, 0.10153866, 0.00825205,
#  0.00168479, 0.06111166, 0.00276793, 0.00315923, 0.00696321, 0.00166403,
#  0.0037369])
# # hist2 = np.array([0.15393065, 0.12219848, 0.19141214, 0.03757275, 0.03137845, 0.04423565,
# #  0.00857152, 0.01437508, 0.14521885, 0.01522797, 0.10153866, 0.00825205,
# #  0.00668479, 0.06111166, 0.00776793, 0.00815923, 0.01696321, 0.01166403,
# #  0.0137369])
# ratio = ((hist2-hist1)/hist1) * 100
# print(np.sum(hist2))

# # 绘制图像
# fig, ax1 = plt.subplots()
# fig.set_size_inches(10, 6)
# plt.set_cmap('RdBu')
# x = np.arange(len(binRange)-1)*5
# # ----------
# # 绘制折线图
# # ----------
# w=1
# ax1.plot(x, ratio, color='#1b71f1', linewidth=5, marker='o',  markersize=17, linestyle='dashed')
# # ax1.plot([x[0]-5*w, x[-1]+5*w], [0.2, 0.2], color='navy', lw=10, linestyle='-', alpha=0.3) # 这是一条直线
# # ax1.plot([x[0]-5*w, x[-1]+5*w], [0.2, 0.2], color='black', lw=1, linestyle='--', alpha=0.3) # 这是一条直线
# ax1.yaxis.set_tick_params(labelsize=15) # 设置y轴的字体的大小
# ax1.set_ylabel("Ratio", fontsize='xx-large')
# ax1.set_ylim(-60,450)
# # ----------
# # 绘制柱状图
# # ----------
# ax2 = ax1.twinx() # 次坐标
# ax2.bar(x-w, hist1, width = 2*w, align='center', alpha=0.5)
# ax2.bar(x+w, hist2, width = 2*w, align='center', alpha=0.5)
# # 设置坐标轴的标签
# ax2.yaxis.set_tick_params(labelsize=15) # 设置y轴的字体的大小
# ax2.set_xticks(x) # 设置xticks出现的位置
# # 创建xticks
# xticksName = []
# for i in range(len(binRange)-1):
#     xticksName = xticksName + ['{}<x<{}'.format(str(np.round(binRange[i],1)), str(np.round(binRange[i+1],1)))]
# ax1.set_xticklabels(xticksName)
# # 设置坐标轴名称
# ax2.set_ylabel("Count", fontsize='xx-large')
# # 设置标题
# # ax2.set_title('The Distribution of Normal1 Data and Normal2 Data', fontsize='x-large')
# # 设置图例
# plt.legend(('Normal1','Nomral2'),fontsize = 'x-large', loc='upper right')
# plt.show()

#######################################################################################################################################################

# # epoch,acc,loss,val_acc,val_loss
# x_axis_data = ['4','8','16','32','64']
# y_axis_data1 = [58.01,58.94,58.65,58.24,57.81]
# y_axis_data2 = [56.17,57.13,56.88,56.09,55.79]
# # y_axis_data3 = [48.74, 50.21, 51.73, 52.41, 52.87, 53.38]
# # complete = [49.01, 51.69, 52.37, 53.18, 53.74, 54.17]

# # 画图
# plt.plot(x_axis_data, y_axis_data1, color='#FF1493', marker='o', alpha=0.8, linewidth=2, label='5% annotation budget')  # '
# plt.plot(x_axis_data, y_axis_data2, color='#1E90FF', marker='^', alpha=0.8, linewidth=2, label='1% annotation budget')
# # plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='w/ ICDSS')
# # plt.plot(x_axis_data, complete, '^--', alpha=0.5, linewidth=1, label='Complete')

# ## 设置数据标签位置及大小
# for a, b in zip(x_axis_data, y_axis_data1):
#     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b1 in zip(x_axis_data, y_axis_data2):
#     plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
# # for a, b2 in zip(x_axis_data, y_axis_data3):
# #     plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
# # for a, b3 in zip(x_axis_data, complete):
# #     plt.text(a, b3, str(b3), ha='center', va='bottom', fontsize=8)
# plt.legend()  # 显示上面的label

# plt.xlabel('N-neighbors size')
# plt.ylabel('mIoU')

# # plt.ylim(-1,1)#仅设置y轴坐标范围
# plt.show()