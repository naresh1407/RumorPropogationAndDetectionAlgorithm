import matplotlib.pyplot as plt
import numpy as np

N = 5
ind = np.arange(N)
width = 0.15
linewidth = 2
fs = 16
fig, ax = plt.subplots(1, 6, sharex=False, sharey=False, figsize=(20, 3))


# BA(100)
x1 = [0.02, 0.04, 0.06, 0.08, 0.1]
y1 = [2.61, 2.56, 2.44, 1.96, 2.3]
# plotting the line 1 points
ax[0].plot(x1, y1, color='cornflowerblue', linestyle='dashed', linewidth=linewidth,
           marker='p', markerfacecolor='cornflowerblue', markersize=10, label='PTVA')
# DL
x2 = x1
y2 = [4.34, 2.18, 2, 1.96, 1.94]
# plotting the points
ax[0].plot(x2, y2, color='lime', linestyle='dashed', linewidth=linewidth,
           marker='x', markerfacecolor='lime', markersize=10, label='GMLA')
# FB
x3 = x1
y3 = [1.80, 1.72, 1.29, 0.75, 0.57]
# plotting the points
ax[0].plot(x3, y3, color='chocolate', linestyle='dashed', linewidth=linewidth,
           marker='*', markerfacecolor='chocolate', markersize=10, label='MLM')
# ax[0][0].set_xticks([0, 1, 2, 3, 4, 5, 6])

ax[0].grid(True)
ax[0].set_xticks([0.02, 0.04, 0.06, 0.08, 0.1])
ax[0].grid(True)
ax[0].set_title('BA network (100)', fontsize=fs)
ax[0].set_yticks([1, 2, 3, 4, 5])
ax[0].set_yticklabels([1, 2, 3, 4, 5], size=fs)
ax[0].set_xticklabels(['0.002', '0.004', '0.006', '0.008', '0.01'], size=15)
# ER(100)
x4 = x1
y4 = [5.1, 3.2, 2.54, 2.4, 2.12]
# plotting the points
ax[1].plot(x4, y4, color='cornflowerblue', linestyle='dashed', linewidth=linewidth,
           marker='p', markerfacecolor='cornflowerblue', markersize=10, label='PTVA')
# BA
x5 = x1
y5 = [5.53, 3.06, 2.32, 2.2, 2.14]
# plotting the points
ax[1].plot(x5, y5, color='lime', linestyle='dashed', linewidth=linewidth,
           marker='x', markerfacecolor='lime', markersize=10, label='GMLA')
# ER
x6 = x1
y6 = [1.00, 0.85, 0.73,  0.61, 0.54]
# plotting the points
ax[1].plot(x6, y6, color='chocolate', linestyle='dashed', linewidth=linewidth,
           marker='*', markerfacecolor='chocolate', markersize=10, label='MLM')
ax[1].set_xticks([0.02, 0.04, 0.06, 0.08, 0.1])
ax[1].grid(True)
ax[1].set_title('ER network (100)', fontsize=fs)
ax[1].set_yticks([1, 2, 3, 4, 5, 6])
ax[1].set_yticklabels([1, 2, 3, 4, 5, 6], size=fs)
ax[1].set_xticklabels(['0.002', '0.004', '0.006', '0.008', '0.01'], size=15)

# BA(500)
x1 = [0.02, 0.04, 0.06, 0.08, 0.1]
y1 = [5.9, 3.87, 3.6, 2.98, 2.94]
# plotting the line 1 points
ax[2].plot(x1, y1, color='cornflowerblue', linestyle='dashed', linewidth=linewidth,
           marker='p', markerfacecolor='cornflowerblue', markersize=10, label='PTVA')
# DL
x2 = x1
y2 = [6.2, 4.2, 3.44, 3, 2.84]
# plotting the points
ax[2].plot(x2, y2, color='lime', linestyle='dashed', linewidth=linewidth,
           marker='x', markerfacecolor='lime', markersize=10, label='GMLA')
# FB
x3 = x1
y3 = [1.17, 0.93, 0.72, 0.78, 0.62]
# plotting the points
ax[2].plot(x3, y3, color='chocolate', linestyle='dashed', linewidth=linewidth,
           marker='*', markerfacecolor='chocolate', markersize=10, label='MLM')
# ax[1][0].set_xticks([0, 1, 2, 3, 4, 5, 6])
# ax[1][0].set_yticks([1, 2, 3, 4, 5])
ax[2].grid(True)
ax[2].set_title('BA network (500)', fontsize=fs)
# ax[1][0].set_yticklabels([1, 2, 3, 4, 5, 6], size=12)
ax[2].set_xticks([0.02, 0.04, 0.06, 0.08, 0.1])
ax[2].set_yticks([1, 2, 3, 4, 5, 6])
ax[2].set_yticklabels([1, 2, 3, 4, 5, 6], size=fs)
ax[2].set_xticklabels(['0.002', '0.004', '0.006', '0.008', '0.01'], size=15)
# ax[1][0].set_title('BA network', fontsize=16)


# ER(500)
x4 = x1
y4 = [4.42, 4.18, 3.26, 3.12, 3.24]
# plotting the points
ax[3].plot(x4, y4, color='cornflowerblue', linestyle='dashed', linewidth=linewidth,
           marker='p', markerfacecolor='cornflowerblue', markersize=10, label='PTVA')
# BA
x5 = x1
y5 = [6.68, 3.06, 2.62, 2.52, 2.52]
# plotting the points
ax[3].plot(x5, y5, color='lime', linestyle='dashed', linewidth=linewidth,
           marker='x', markerfacecolor='lime', markersize=10, label='GMLA')
# ER
x6 = x1
y6 = [0.93, 0.69, 0.67, 0.74, 0.59]
# plotting the points
ax[3].plot(x6, y6, color='chocolate', linestyle='dashed', linewidth=linewidth,
           marker='*', markerfacecolor='chocolate', markersize=10, label='MLM')
# ax[1][1].set_yticks([2, 3, 4, 5, 6])
ax[3].set_xticks([0.02, 0.04, 0.06, 0.08, 0.1])
ax[3].grid(True)
ax[3].set_yticks([1, 2, 3, 4, 5, 6, 7])
ax[3].set_yticklabels([1, 2, 3, 4, 5, 6, 7], size=fs)
ax[3].set_title('ER network (500)', fontsize=fs)
ax[3].set_xticklabels(['0.002', '0.004', '0.006', '0.008', '0.01'], size=15)
# ax[1][1].set_title('ER network', fontsize=16)

# BA(1000)
x1 = [0.02, 0.04, 0.06, 0.08, 0.1]
y1 = [5.17, 4.28, 3.6, 3.16, 3.02]
# plotting the line 1 points
ax[4].plot(x1, y1, color='cornflowerblue', linestyle='dashed', linewidth=linewidth,
           marker='p', markerfacecolor='cornflowerblue', markersize=10, label='PTVA')
# DL
x2 = x1
y2 = [5.5, 3.1, 2.98, 2.7, 2.65]
# plotting the points
ax[4].plot(x2, y2, color='lime', linestyle='dashed', linewidth=linewidth,
           marker='x', markerfacecolor='lime', markersize=10, label='GMLA')
# FB
x3 = x1
y3 = [0.67, 0.44, 0.48, 0.35, 0.25]
# plotting the points
ax[4].plot(x3, y3, color='chocolate', linestyle='dashed', linewidth=linewidth,
           marker='*', markerfacecolor='chocolate', markersize=10, label='MLM')
ax[4].set_xticks([0.02, 0.04, 0.06, 0.08, 0.1])
# ax[2][0].set_yticks([1, 2, 3, 4, 5])

ax[4].grid(True)
ax[4].set_title('BA network (1000)', fontsize=fs)
ax[4].set_yticks([1, 2, 3, 4, 5, 6])
ax[4].set_yticklabels([1, 2, 3, 4, 5, 6], size=fs)
# ax[4].set_yticklabels([1, 2, 3, 4, 5, 6], size=12)
ax[4].set_xticklabels(['0.002', '0.004', '0.006', '0.008', '0.01'], size=15)
# ax[2][0].set_title('BA network', fontsize=16)


# ER(1000)
x4 = x1
y4 = [5.3, 4.92, 3.84, 3.46, 3.22]
# plotting the points
ax[5].plot(x4, y4, color='cornflowerblue', linestyle='dashed', linewidth=linewidth,
           marker='p', markerfacecolor='cornflowerblue', markersize=10, label='PTVA')
# BA
x5 = x1
y5 = [5.78, 3.89, 3.72, 3.6, 3.1]
# plotting the points
ax[5].plot(x5, y5, color='lime', linestyle='dashed', linewidth=linewidth,
           marker='x', markerfacecolor='lime', markersize=10, label='GMLA')
# ER
x6 = x1
y6 = [0.95, 0.73, 0.65, 0.63, 0.63]
# plotting the points
ax[5].plot(x6, y6, color='chocolate', linestyle='dashed', linewidth=linewidth,
           marker='*', markerfacecolor='chocolate', markersize=10, label='MLM')
ax[5].set_xticks([0.02, 0.04, 0.06, 0.08, 0.1])
ax[5].set_yticks([1, 2, 3, 4, 5, 6])
ax[5].set_yticklabels([1, 2, 3, 4, 5, 6], size=fs)
# ax[2][0].set_yticks([1, 2, 3, 4, 5])
# ax[2][1].set_yticks([2, 3, 4, 5, 6])

ax[5].grid(True)
ax[5].set_title('ER network (1000)', fontsize=fs)
ax[5].set_xticklabels(['0.002', '0.004', '0.006', '0.008', '0.01'], size=15)
# ax[2][1].set_title('ER network', fontsize=16)


fig.text(0.5, 0.02, 'Density of network', ha='center', fontsize=fs)
fig.text(0.001, 0.5, 'Distance error', va='center',
         rotation='vertical', fontsize=fs)
# plt.tight_layout()
plt.legend()
# function to show the plot
plt.tight_layout()
plt.show()
plt.savefig("scatter_graph_dog_vs_de_2.eps", dpi=1200)
