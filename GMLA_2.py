# Algorithms used for rumor source inference in a graph
# - Pinto algorithm

import networkx as nx
import numpy
# from numpy import matrix, array
# from numpy import linalg
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import json
# from numba import jit
import plotting


def observed_delay(g, O):
    d = numpy.zeros(shape=(len(O) - 1, 1))
    for i in range(len(O) - 1):
        d[i][0] = g.nodes[O[i + 1]]['time'] - g.nodes[O[i]]['time']
    return d


def delay_covariance(T, O, sigma2):
    """
    Computes lambda
    :param T: tree
    :param O: list of observers
    :param sigma2: variance
    :return:
    """
    # TODO stop using all_simple_paths (complexity)
    n = len(O)
    # print("n", n)
    delta = numpy.zeros(shape=(n - 1, n - 1))
    # print(delta.shape)
    T = T.to_undirected()
    # print("T", T)
    for k in range(n - 1):
        for i in range(n - 1):
            if i == k:
                delta[k][i] = len(
                    list(nx.all_simple_paths(T, O[0], O[k + 1]))[0]) - 1
            else:
                # print("k", k, "i", i)
                c1 = list(nx.all_simple_paths(T, O[0], O[k + 1]))[0]
                c2 = list(nx.all_simple_paths(T, O[0], O[i + 1]))[0]
                # print(c2)
                S = [x for x in c1 if x in c2]
                delta[k][i] = len(S) - 1
    delta = delta * (sigma2 ** 2)  # FIXME : square or not ?
    return delta


def deterministic_delay(T, s, O, mi):
    """
    Computes mu_s
    :param T: tree
    :param s: source
    :param O: list of observers
    :param mi: mean
    :return:
    """
    constant = height_node(T, s, O[0])
    mi_s = numpy.zeros(shape=(len(O) - 1, 1))
    for i in range(len(O) - 1):
        mi_s[i][0] = height_node(T, s, O[i + 1]) - constant
    mi_s = mi * mi_s
    return mi_s


def height_node(T, s, node):
    l = list(nx.all_simple_paths(T, s, node))
    if l == []:
        return 0
    else:
        return len(l[0]) - 1


# @jit(parallel=True)
def run(g, O, mi, sigma2, O_length, d):
    """
    Main
    :param g: graph
    :param O: list of observers <<<< ACTIVE observers !
    :param mi: mean
    :param sigma2: variance
    :return:
    """
    # TODO : consider only active observers !
    first_node = O[0]

    # Compute the delay vector d relative to first_node

    print("observed dalay completed")
    # Score list of pair (node , score)
    s = {}
    # s=[(O[0],0)]

    # initialise (node , score)
    v = [O[0], 0]

    flag = 0

    # max score
    s_max = 0

    while v[1] >= s_max:
        Tv = {}
        Tv[v[0]] = 0
        for n in g[v[0]]:
            if n not in s.keys():
                tree_bfs = nx.bfs_tree(g, source=first_node)
                mu_s = deterministic_delay(tree_bfs, n, O, mi)
                delta = delay_covariance(tree_bfs, O, sigma2)
                inverse = numpy.linalg.inv(delta)
                score = (numpy.exp(-.5 * numpy.dot(numpy.dot((d - mu_s).T, inverse), (d - mu_s)))) / (
                    numpy.sqrt(abs(numpy.linalg.det(delta))))
                Tv[n] = score[0][0]
        if len(Tv) != 0:
            neighbor_n = [a for a in g[v[0]]]

            if v[0] not in Tv:
                curr_score_node = s[v[0]]
            else:
                curr_score_node = Tv[v[0]]
            curr_node = v[0]

            change = 0

            for a in neighbor_n:
                if a in Tv:
                    neigh_score_a = Tv[a]
                else:
                    continue

                if curr_score_node < neigh_score_a:
                    change = 1
                    curr_node = a

            if change != 0:
                s.update(Tv)
                v = [curr_node, s[curr_node]]
                s_max = v[1]

            else:
                break

        # end of if
    # end of while

    sorted_score = sorted(s.items(), key=lambda kv: kv[1], reverse=True)

    print("The rumour source in order : ")
    for a in range(1):
        print(sorted_score[a][0], "\t")
        infected_node1 = sorted_score[a][0]
    return s, infected_node1


def from_pandas_dataframe(df, source, target, edge_attr=None, create_using=None):
    g = nx.Graph()

    # Index of source and target
    src_i = df.columns.get_loc(source)
    tar_i = df.columns.get_loc(target)
    if edge_attr:
        # If all additional columns requested, build up a list of tuples
        # [(name, index),...]
        if edge_attr is True:
            # Create a list of all columns indices, ignore nodes
            edge_i = []
            for i, col in enumerate(df.columns):
                if col is not source and col is not target:
                    edge_i.append((col, i))
        # If a list or tuple of name is requested
        elif isinstance(edge_attr, (list, tuple)):
            edge_i = [(i, df.columns.get_loc(i)) for i in edge_attr]
        # If a string or int is passed
        else:
            edge_i = [(edge_attr, df.columns.get_loc(edge_attr)), ]

        # Iteration on values returns the rows as Numpy arrays
        for row in df.values:
            g.add_edge(row[src_i], row[tar_i], {i: row[j] for i, j in edge_i})

    # If no column names are given, then just return the edges.
    else:
        for row in df.values:
            g.add_edge(row[src_i], row[tar_i])
    return g


def sensor_node_selection(g1):
    sensor_nodes1 = []
    for sen1 in g1.nodes():
        count4 = int(sen1 % 30)
        if count4 == 0:
            sensor_nodes1.append(int(sen1))
            # print("sen", sen)
    return sensor_nodes1


####################################################################################

# numpy.random.seed(52)  # option for reproducibility---- 51,52,53,54,55,57,58,59
total_distance = 0
repeat = 50
repeat_list = []
error_distance = []
time_list = []
total_time = 0
filename = "dolphins"
df = pd.read_csv(filename + '.csv', delimiter=',')
df = df[["Source", "Target"]]
g = from_pandas_dataframe(df, source='Source', target='Target')

node_len = len(g.nodes)
min_node = min(g.nodes)
max_node = max(g.nodes)
sorted_nodes = sorted(g.nodes)
length_between_nodes = dict(nx.all_pairs_shortest_path_length(g))

# k0 = math.ceil(math.sqrt(len(g)))
k0 = int(len(g.nodes)/20)
# k0 = 15
for rep in range(1, repeat + 1):
    # ############################################################ Change here

    # print("k0", k0)
    O = np.random.choice(g.nodes, k0, replace=False).tolist()
    O_length = len(O)
    # ###################################################################################
    # O = numpy.random.randint(low=min_node, high=max_node, size=k0).tolist()
    t = numpy.random.uniform(low=0.000000001, high=6, size=k0).tolist()
    print("len of g nodes", len(g.nodes))
    print("density", nx.density(g))
    print("observers length", O_length)
    # print("t = ", t)
    # print("O = ", O)
    start = time.time()
    for a in range(0, O_length):
        # print(t[a])
        g.nodes[O[a]]['time'] = t[a]
        # print(O[a], ":", g.nodes[O[a]])
    mi = numpy.mean(t)
    sigma2 = numpy.var(t)

    # #####Running the algorithm:
    # run(g, O, mi, sigma2,k0)
    d = observed_delay(g, O)
    score, infected_nodes = run(g, O, mi, sigma2, O_length, d)
    end = time.time()
    total_time = total_time + (end - start)
    time_list.append(total_time)
    min_time = min(t)
    min_index = t.index(min_time)
    # print(min_index, min_time)
    infected_node = O[min_index]
    print("infected_node", infected_node)
    infected_nodes_and_shortest_path = length_between_nodes[infected_nodes]
    # print("infected_nodes_and_shortest_path", infected_nodes_and_shortest_path)
    distance = infected_nodes_and_shortest_path.get(infected_node)
    print("distance error\t", distance, "\n")
    error_distance.append(distance)
    total_distance = total_distance + distance

    repeat_list.append(rep)
    print("avg error distance is: ", total_distance/len(repeat_list))
    print("############################################## repeating ", rep, "time")
print("error_distance", error_distance)
print(repeat_list)
error_distance_dict = dict(zip(repeat_list, error_distance))
total_time_dict = dict(zip(repeat_list, time_list))
print(error_distance_dict)
with open("GMLA_" + filename + "_" + str(O_length) + "_error_distance.json", "w") as a:
    json.dump(error_distance_dict, a)

with open("GMLA_" + filename + "_" + str(O_length) + "_total_time.json", "w") as outfile:
    json.dump(total_time_dict, outfile)

print("avg distance error: ", total_distance / repeat)
print("avg computation time : ", total_time / repeat)
score = sorted(score.items(), key=lambda kv: kv[1], reverse=True)

# ################################# Coloring nodes
node_color = []
count = 0
score_list = []
count2 = 0
for a, b in enumerate(score):
    # print(a, b)
    if count2 < 1:
        score_list.append(b[0])
        count2 = count2 + 1

# print("score_list", score_list)
for node in g.nodes():
    # if the node has the attribute group1
    # elif list(node)[0] in score[:5]:

    if node in score_list:
        node_color.append('red')
    elif node in O:
        node_color.append('green')
    else:
        node_color.append('blue')
nx.draw(g, node_size=300, node_color=node_color,
        alpha=1, linewidths=0.5, width=0.5, edge_color='black', with_labels=True)
# plt.savefig(filename + "_PTVA.png")
zeros = []
ones = []
twos = []
threes = []
fours = []
fives = []
sixes = []
for i in error_distance:
    if i == 0:
        zeros.append(i)
    if i == 1:
        ones.append(i)
    if i == 2:
        twos.append(i)
    if i == 3:
        threes.append(i)
    if i == 4:
        fours.append(i)
    if i == 5:
        fives.append(i)
    if i == 6:
        sixes.append(i)
print("0's:", len(zeros), ", 1's:", len(ones), ", 2's:", len(twos), ", 3's:", len(
    threes), ", 4's:", len(fours), ",5's:", len(fives), ", 6's:", len(sixes))
# print("Runtime of the program is", {end - start})
plt.show()
