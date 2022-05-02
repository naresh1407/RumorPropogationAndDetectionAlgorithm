import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from matplotlib.lines import Line2D
import time


# def accuracy(source_node, predicted_set):
#     a = []
#     for i in source_node:
#         if i in predicted_set:
#             a.append(1 / (predicted_set[0]))
#         else:
#             a.append(0)
#     acc = np.mean(a)
#     return acc


def load(filename):
    df = pd.read_csv(filename + '.csv')
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df, source='Source', target='Target', create_using=Graphtype)
    # nx.draw(G, with_labels=True)
    # # G = nx.karate_club_graph()
    # plt.clf()
    # nx.draw(G, with_labels=True)
    # plt.savefig(filename + 'PTVA.png')

    return G, len(G)


def infect_graph(g, filename):
    """
    Function to infect the graph using SI model.
    Parameters:
      g: Graph
    Returns:
      G : Infected graph
      t : Time of diffusion of each node
    """
    G = g
    # Model selection - diffusion time
    model = ep.SIModel(G)
    nos = 1 / len(G)
    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', 0.03)
    config.add_model_parameter("fraction_infected", 1/len(G))
    model.set_initial_status(config)

    # Simulation execution
    iterations = model.iteration_bunch(200)

    diffusionTime = {}
    for i in range(1, len(G)):
        diffusionTime[i] = -1

    for i in iterations:
        for j in i['status']:
            if i['status'][j] == 1:
                diffusionTime[j] = i['iteration']

    nodeColor = []
    source_nodes = []
    for i in G.nodes():
        if iterations[0]["status"][i] == 1:
            # nodeColor.append('red')
            source_nodes.append(i)
        # else:
        #     nodeColor.append('blue')
    print("source nodes", source_nodes)
    sorted_values = sorted(diffusionTime.values())  # Sort the values
    # print(sorted_values)
    sorted_dict = {}

    for i in sorted_values:
        for k in diffusionTime.keys():
            if diffusionTime[k] == i:
                sorted_dict[k] = diffusionTime[k]
    # print("sorted_dict", sorted_dict)
    plt.clf()
    # nx.draw(G, node_color=nodeColor, with_labels=True)
    # plt.filename('Intial Phase')
    # plt.savefig(f'{filename}_Initial-infect.png')
    # plt.clf()
    # nx.draw(G, node_color=list(x for i, x in diffusionTime.items()), cmap=plt.cm.Reds, with_labels=True)
    # plt.show()
    # plt.savefig(filename +"_Final-infect.png")

    return G, sorted_dict, source_nodes


_legends = [Line2D([0], [0], marker='o', color='w', label='Source', markerfacecolor='r', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Observers', markerfacecolor='g', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Others', markerfacecolor='b', markersize=15), ]

node_color = []


def coloring(filename, G, nodes, score_list, O, algo, sourceNodes):
    for i in nodes:
        if i in score_list:
            node_color.append('red')
        elif i in sourceNodes:
            node_color.append('yellow')
        elif i in O:
            node_color.append('green')
        else:
            node_color.append('blue')

    plt.clf()
    nx.draw(G, with_labels=True, node_size=300, node_color=node_color)
    plt.legend(_legends, ['Source', 'Observers', 'Others'], loc="upper right")
    plt.savefig(filename + "_algo_ptva.png")


# helpers ----------------------------------------------------------------
def delayVector(G, t1, observers):
    """Calcs delay with respect to the t1"""

    d = np.zeros(shape=(len(observers) - 1, 1))

    for i in range(O_length-1):
        # print(i+1)
        # print("\t\t\t\t", G.nodes[observers[i]]['time'])
        d[i][0] = G.nodes[observers[i + 1]]['time'] - t1
    return d


def nEdges(bfs, s, a):
    """ Returns list of edges from s -> a"""
    try:
        l = list(nx.all_simple_paths(bfs, s, a))[0]
        return l
    except:
        return [0]


def intersection(l1, l2):
    temp = set(l2)
    l = [x for x in l1 if x in temp]
    return len(l) - 1


def mu(bfs, mn, observers, s):
    """Calcs the determinticDelay w.r.t bfs tree."""

    o1 = observers[0]
    length_o1 = len(nEdges(bfs, s, o1))
    mu_k = np.zeros(shape=(len(observers) - 1, 1))
    for k in range(len(observers) - 1):
        mu_k[k][0] = len(nEdges(bfs, s, observers[k + 1])) - length_o1

    return np.dot(mu_k, mn)


def covariance(bfs, sigma2, observers, s):
    """Cals the delayCovariance of the bfs tree."""

    o1 = observers[0]
    delta_k = np.zeros(shape=(len(observers) - 1, len(observers) - 1))
    for k in range(len(observers) - 1):
        for i in range(len(observers) - 1):
            if k == i:
                delta_k[k][i] = len(nEdges(bfs, o1, observers[k + 1]))
            else:
                ne1 = nEdges(bfs, o1, observers[k + 1])
                ne2 = nEdges(bfs, o1, observers[i + 1])
                delta_k[k][i] = intersection(ne1, ne2)

    return sigma2 * delta_k


# Main Algo ----------------------------------------------------------------
def PTVA(G, observers, Ka, sigma2, mn):
    """Main Function for PTVA"""

    # selecting t1
    t1 = G.nodes[observers[0]]['time']
    print("t1", t1)
    # computing d
    d = delayVector(G, t1, observers)

    # score
    likelihood = {}

    for s in list(G.nodes()):
        bfs = nx.bfs_tree(G, source=s)
        mu_s = mu(bfs, mn, observers, s)
        delta_s = covariance(bfs, sigma2, observers, s)
        score = np.dot(np.dot(mu_s.T, np.linalg.inv(delta_s)), d - (0.5) * mu_s)
        likelihood[s] = score

    sortedLikelihood = sorted(likelihood.items(), key=lambda x: x[1], reverse=True)
    count = 0
    for i, x in sortedLikelihood:
        if count < 5:
            print(f'The node {i} has {x[0][0]} likelihood')
        count = count + 1
    return sortedLikelihood


def sensor_node_selection(g1):
    sensor_nodes1 = []
    for sen1 in g1.nodes():
        count4 = int(sen1 % 20)
        if count4 == 0:
            sensor_nodes1.append(int(sen1))
            # print("sen", sen)
    return sensor_nodes1


# Main Part -----------------------------------------------------------------
repeat = 50
error_distance = []
total_time = 0
total_distance = 0
time_list = []
filename = 'football'
algo = 'ptva'
G, _ = load(filename)
length_between_nodes = dict(nx.all_pairs_shortest_path_length(G))
c = 0
for i in range(repeat):
    print("############################### repeating", i,
          "time ###############################")
    # Infect graph
    G, arrivalTime, sourceNodes = infect_graph(G, filename=filename)
    print(len(G.nodes))
    # Take observers
    start = time.time()
    # k0 = math.ceil(math.sqrt(len(G)))
    # print("k0", k0)
    k0 = int(len(G.nodes)/20)
    # k0 = 8
    np.random.seed(k0)
    all_observers = np.random.choice(G.nodes, k0, replace=False).tolist()
    print("observers length", len(all_observers))
    observers = []
    for i in all_observers:
        if arrivalTime[i] != -1:
            observers.append(i)
    # print("observers", observers)
    # observers = sensor_node_selection(G)
    O_length = len(observers)
    print("o length", O_length)
    # mean and variance
    t = []

    for i in observers:
        # print(i)
        if arrivalTime[i] != -1:
            # print("\t", i)
            t.append(arrivalTime[i])
    mn = np.mean(t)
    sigma2 = np.var(t)
    # print(t)
    # print("t", len(t))
    # assigning time attr to each node.
    for i in range(0, O_length):
        # print("i", i)
        # print("t[i]", t[i])
        G.nodes[observers[i]]['time'] = t[i]
        # print(observers[i], ":", G.nodes[observers[i]])

    score = PTVA(G, observers, k0, sigma2, mn)

    scoreList = [score[i][0] for i in range(5)]
    nodes = [list(a)[0] for a in G.nodes(data=True)]

    print()
    print(f'Sources : {sourceNodes}')
    print(f'Predicted : {scoreList}')
    # print(f'accuracy : {accuracy(sourceNodes, scoreList)} %')

    end = time.time()
    time_1 = end - start
    time_list.append(time_1)
    print("Runtime of the program is", {end - start})
    # coloring(filename, G, nodes, scoreList, observers, algo, sourceNodes)
    total_time = time_1 + total_time
    infected_nodes_and_shortest_path = []
    for key, value in length_between_nodes.items():
        if sourceNodes[0] == key:
            infected_nodes_and_shortest_path.append(dict(sorted(value.items(), key=lambda z: z[1])))

    for i, dictionary in enumerate(infected_nodes_and_shortest_path):
        for node, distance in dictionary.items():
            if scoreList[0] == node:
                print("distance error\t", distance, "\n")
                error_distance.append(distance)
                total_distance = total_distance + distance
    c = c+ 1
    print("avg distance error is : ", total_distance/c)
avg_time = total_time/repeat
avg_distance = total_distance/repeat
print("avg_time", avg_time)
print("avg_distance", avg_distance)
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
print("0's:", len(zeros), ", 1's:", len(ones), ", 2's:", len(twos), ", 3's:", len(threes), ", 4's:", len(fours), ", 5's:", len(fives), ", 6's:", len(sixes))
# plt.show()
