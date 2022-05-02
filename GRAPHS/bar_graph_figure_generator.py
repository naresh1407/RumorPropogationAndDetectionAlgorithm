import numpy as np
import matplotlib.pyplot as plt

N = 5
ind = np.arange(N)
width = 0.15
fig, ax = plt.subplots(1, 4, sharex=False, sharey=False, figsize=(12, 3))
xvals = [6, 24, 52, 18, 0]
ax[0].bar(ind, xvals, width, color='cornflowerblue', label='PTVA')

yvals = [4, 30, 46, 20, 0]
ax[0].bar(ind + width, yvals, width, color='lime', label='GMLA')

zvals = [3, 23, 55, 19, 0]
ax[0].bar(ind + width * 2, zvals, width, color='chocolate', label='MLM')

ax[0].grid(True)
ax[0].legend()
# plt.xlabel("Distance error")
# plt.ylabel('Frequency [%]')
# ax[1].title("KT network")

ax[0].set_xticks([0, 1, 2, 3, 4])
# ax[0].set_xticklabels(['0', '1', '2', '3', '4'], size=12)
# ax[0].set_yticklabels(['0', '10', '20', '30', '40', '50', '60'], size=12)
ax[0].set_title('KT network', fontsize=12)
# plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))
# plt.savefig("bar_graph_DDE_KT")
# plt.show()

N = 7
ind = np.arange(N)

xvals = [0, 8, 16, 8, 52, 16, 0]
ax[1].bar(ind, xvals, width, color='cornflowerblue', label='PTVA')

yvals = [0, 16, 30, 26, 14, 6, 4]
ax[1].bar(ind + width, yvals, width, color='lime', label='GMLA')

zvals = [5, 26, 40, 22, 7, 0, 0]
ax[1].bar(ind + width * 2, zvals, width, color='chocolate', label='MLM')

ax[1].grid(True)
ax[1].legend()
# plt.xlabel("Distance error")
# plt.ylabel('Frequency [%]')
# plt.title("DL network")

ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6])
# ax[1].set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], size=12)
ax[1].set_title('DL network', fontsize=12)
# plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))

# plt.savefig("bar_graph_DDE_DL")
# plt.show()

# plt.subplot(1, 4, 3, sharey=True)
N = 5
ind = np.arange(N)

xvals = [0, 18, 42, 40, 0]
ax[2].bar(ind, xvals, width, color='cornflowerblue', label='PTVA')

yvals = [0, 14, 42, 42, 2]
ax[2].bar(ind + width, yvals, width, color='lime', label='GMLA')

zvals = [2, 36, 58, 4, 0]
ax[2].bar(ind + width * 2, zvals, width, color='chocolate', label='MLM')

ax[2].grid(True)
ax[2].legend()
# plt.xlabel("Distance error")
# plt.ylabel('Frequency [%]')
# plt.title("FL network")
# #
ax[2].set_xticks([0, 1, 2, 3, 4])
# ax[2].set_xticklabels(['0', '1', '2', '3', '4'], size=12)
# ax[2].set_yticklabels(['0', '10', '20', '30', '40', '50', '60'], size=14)
ax[2].set_title('FL network', fontsize=12)
# plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))
# plt.savefig("bar_graph_DDE_FL")
# plt.show()

N = 7
ind = np.arange(N)

xvals = [0, 10, 26, 26, 22, 16, 0]
ax[3].bar(ind, xvals, width, color='cornflowerblue', label='PTVA')

yvals = [0, 12, 34, 36, 8, 10, 0]
ax[3].bar(ind + width, yvals, width, color='lime', label='GMLA')

zvals = [0, 34, 40, 16, 10, 0, 0]
ax[3].bar(ind + width * 2, zvals, width, color='chocolate', label='MLM')

ax[3].grid(True)
# plt.xlabel("Distance error")
# plt.ylabel('Frequency [%]')
# plt.title("FB1 network")

ax[3].set_xticks([0, 1, 2, 3, 4, 5, 6])
# ax[3].set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], size=12)
ax[3].legend()
ax[3].set_title('FB1 network', fontsize=12)
# plt.savefig("bar_graph_DDE_FB1")
# plt.xlabel('Distance error')
# plt.ylabel('Frequency [%]')
fig.text(0.5, 0.01, 'Distance error', ha='center', fontsize=12)
fig.text(0.001, 0.5, 'Frequency [%]',
         va='center', rotation='vertical', fontsize=12)

plt.tight_layout()
plt.savefig("bar_graph_combined_dde_with_titles_2_2.eps", dpi=1200)
plt.show()
