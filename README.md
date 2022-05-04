# RumorPropogationAndDetectionAlgorithm

## Implementation flow:
* Read the graph in the edgelist fashion using csv files.
* For each edge, randomly assign the time delay for information propagation between these
nodes.
* Now construct the graph in adjacency list fashion from the edge list available.
* Choose a random set of sources.
* Propagate the information across all the connected nodes using dijkstra’s algorithm.
* Now choose a random set of nodes as the observers.
* Now use the infomed times of these nodes to compute the sources using dijkstra’s
algorithm in a backtracking fashion.

## Analysis Flow:
* For each csv file, iterate the diffusion and localisation models, 50 times.
* For each iteration, record the execution time, distance error and candidate sources.
* After the completion of all the iterations, compute the average execution time, average
distance error and distribution of distance error.

![image](https://user-images.githubusercontent.com/70414698/166830346-8280ed57-0af2-44a7-b855-2a894bf4f8d2.png)

