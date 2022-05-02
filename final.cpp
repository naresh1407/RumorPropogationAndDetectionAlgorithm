#include <bits/stdc++.h>
#include <fstream>
using namespace std;

#define INF (int)(1e9 + 7)
#define int long long int

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
int rand(int l, int r){
  uniform_int_distribution<int> ludo(l, r); return ludo(rng);
}

int vdm; 

// Program to generate the adjacency list graph out of csv files 
void inputGraph(vector<vector<pair<int, int>>> &G, int &n, int &m, vector <int> &nodes, string &fname) {

    int from, to, timeDelay;
    unordered_set <int> uniqueNodes;

    cout << "\n \nProcessing the graph ... \n \n";

    ifstream fileStream(fname);
    vector <pair<int, int>> v;
    int a, b;
    while (fileStream >> a && fileStream >> b) {
        v.push_back({a, b});
        n = max(n, max(a, b));
        uniqueNodes.insert(a);
        uniqueNodes.insert(b);
        m++;
    }

    fileStream.close();

    for(auto x: uniqueNodes) nodes.push_back(x);

    cout << "Max N: " << n << endl;

    //Get no. of edges
    G.resize(n + 7);

    //Create the graph
    for (int i = 0; i < m; i++) {

        to = v[i].first;
        from = v[i].second;
        timeDelay = rand(1LL, 10LL);
        // cout<<from<<" "<<to<<" "<<timeDelay<<"\n";
        G[from].push_back({to, timeDelay});
        G[to].push_back({from, timeDelay});
    }

    cout<<"\n";

    cout << "Graph processed.\n \n";
}

// Function to diffuse the information through a source using Dijkstra's algorithms
void propDijkstra(vector<vector<pair<int,int>>> &G, int source, int n,
    vector <int> &infoTimes, int timeInit, vector <int> &nodes) {

    vector <bool> visited(n+7, false);
    infoTimes[source] = min(infoTimes[source], timeInit);

    for(int iteration = 0; iteration < n; iteration++) {

        int chosenNode = -1;
        for(int currNode = 1; currNode <= n; currNode++) {
            if(!visited[currNode] && (chosenNode == -1 || infoTimes[currNode] < infoTimes[chosenNode])) {
                chosenNode = currNode;
            }
        }

        if(infoTimes[chosenNode] == INF) break;

        visited[chosenNode] = true;

        for(auto edge : G[chosenNode]) {

            int adjacentNode = edge.first;
            int delay = edge.second;

            if(infoTimes[chosenNode] + delay < infoTimes[adjacentNode]) {
                infoTimes[adjacentNode] = min(infoTimes[adjacentNode], infoTimes[chosenNode] + delay);
            }
        }
    }

}

// Information diffusion model function
void propogateGraph(vector <int> &infoTimes, vector <vector<pair<int, int>>> &G, int n, vector <int> &actualSources, vector <int> &nodes) {

    int k = n/10, timeInit = rand(1000, 5000);
    int nodesSize = nodes.size();

    cout << "Initial diffusion time: " << timeInit << endl;
    vector <int> sources;
    cout << "Actual diffusion sources: ";

    unordered_set <int> processed;
    for(int i = 0 ; i < k; i++) {
        int currSource;
        while(1) {
            currSource = rand(0, nodesSize - 1);
            if(processed.find(currSource) != processed.end()) continue;
            cout << currSource << " ";
            processed.insert(currSource);
            sources.push_back(nodes[currSource]);
            break;
        }
        infoTimes[nodes[currSource]] = timeInit;
    }

    cout << endl;
    actualSources = sources;


    cout << "\n Diffusing the information in the graph... \n \n";
    for(auto currSource : sources) {
        propDijkstra(G, currSource, n, infoTimes, timeInit, nodes);
    }
    cout << "\n Information diffused in the graph. \n";

    // cout << "\n Informed times of Nodes in the graph: \n";
    // for(int i = 1; i <= n; i++) cout << infoTimes[i] << " ";
    // cout << endl;

}

// Floyd Warshall algorithm to compute all pair shortest path between 
void floyd(vector <vector<int>> &dp, int n) {
    for(int k = 1; k <= n; k++) {
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= n; j++) {
                if(dp[i][k] < INF && dp[k][j] < INF)
                dp[i][j] = min(dp[i][j], (dp[i][k] + dp[k][j]));
                if(dp[i][j] < INF)  vdm = max( vdm, dp[i][j]);
            }
        }
    }
}


// Helper function to locate the sources using backward propagation and Dijkstra's algorithm
void locDijkstra(vector<vector<int>> &minDelay, vector<vector<pair<int, int>>> &G, vector<int> &infoTimes, int n, int observer, vector <int> &nodes) {

    vector<bool> visited(n + 1, false);
    minDelay[observer][observer] = infoTimes[observer];

    for (int iteration = 0; iteration < n; iteration++)
    {

        int chosenNode = -1;
        for (int currNode = 1; currNode <= n; currNode++)
        {
            if (!visited[currNode] && (chosenNode == -1 || minDelay[observer][currNode] < minDelay[observer][chosenNode]))
            {
                chosenNode = currNode;
            }
        }

        if (minDelay[observer][chosenNode] == INF)
            break;

        visited[chosenNode] = true;

        for (auto edge : G[chosenNode])
        {

            int adjacentNode = edge.first;
            int delay = edge.second;

            if (minDelay[observer][adjacentNode] == INF && (minDelay[observer][chosenNode] - delay < minDelay[observer][adjacentNode]))
                minDelay[observer][adjacentNode] = minDelay[observer][chosenNode] - delay;
        }
    }
}

// Source localisation model function
void locateSources(vector <vector<pair<int, int>>> &G, vector <int> &informedTimes, int n, vector <int> &calcluatedSources,
    vector <vector<int>> &dp, vector <int> &nodes) {

    int observerCount = n/10;
    vector <int> observers;
    vector <vector<int>> minDelay(n + 7, vector<int>(n+7, INF));

    int minObservationTime = INF;
    unordered_set <int> processed;

    int nodesSize = nodes.size();

    cout << "Observers: ";
    for(int i = 0; i < observerCount; i++) {
        int observer;
        while(1) {
            observer = rand(0, nodesSize - 1);
            if(processed.find(observer) != processed.end()) continue;
            processed.insert(observer);
            cout << observer << " ";
            observers.push_back(nodes[observer]);
            break ;
        }
        int  observerNode = nodes[observer];
        minDelay[observerNode][observerNode] = informedTimes[observerNode];
        minObservationTime = min(minObservationTime, informedTimes[observerNode]);
    }
    cout << endl;
    cout << endl;


    // for(int i = 1; i <= n; i++) {
    //     for(auto x: G[i]) {
    //         dp[i][x.first] = x.second;
    //         dp[x.first][i] = x.second;
    //     }
    // }

    cout << "\n Locating the sources... \n \n";
    cout << "\n Computing distances... \n \n";

    // floyd(dp, n);
    // for(auto observer : observers) {
    //     for(int i = 1; i <= n; i++) {
    //         minDelay[observer][i] = minDelay[observer][observer] - dp[observer][i];
    //     }
    // }

    for(auto observer : observers) {
        locDijkstra(minDelay, G, informedTimes, n, observer, nodes);
    }

    cout << "\n Distances computed... \n \n";



    int cnt = 0;
    // for(auto observer: observers) {
    //     cout << "Observer #" << ++cnt << ": " << observer << endl;
    //     for(int i = 1;  i <= n; i++) cout << minDelay[observer][i] << " ";
    //     cout << endl;
    // }

    vector <pair<int, int>> maxDistAndFreq(n+7);
    for(int i = 0; i <=n; i++) {
        maxDistAndFreq[i].first = 0;
        maxDistAndFreq[i].second = 0;
    }

    for(auto observer : observers) {
        for(auto i : nodes) {
            if(minDelay[observer][i] != INF) maxDistAndFreq[i].first = max(maxDistAndFreq[i].first, minDelay[observer][i]);
        }
    }

    int maxFreq = 0;
    for(auto observer : observers) {
        for(auto i : nodes) {
            if(minDelay[observer][i] == maxDistAndFreq[i].first) {
                maxDistAndFreq[i].second++;
                maxFreq = max(maxFreq, maxDistAndFreq[i].second);
            }
        }
    }

    vector <int> sources;
    int timeInit = INF;

    for(auto i : nodes) {

        int relativeTime = maxDistAndFreq[i].first;
        int freq = maxDistAndFreq[i].second;

        if(relativeTime <= minObservationTime && freq == maxFreq) {
            sources.push_back(i);
            timeInit = relativeTime;
        }
    }

    calcluatedSources = sources;

    cout << "\n Sources computed ...\n \n";
    cout << "\n Computer sources size: " << sources.size() << endl;
    cout  << "Initial time: " << timeInit << endl;
}

// Function to compute probable sources in the periphery of the actual source node
vector <int> candidateSources(vector <vector<pair<int, int>>> &G, int n, vector <int> &actualSources, vector <int> &computedSources,
    vector <vector<int>> &dp) {

    int actSz = actualSources.size();
    int cmpSz = computedSources.size();
    int mnSz = min(actSz, cmpSz);
    int mxSz = max(actSz, cmpSz);

    priority_queue <pair<int, pair<int, int>>> pq;
    for(int i = 0; i < cmpSz; i++) {
        for(int j = 0; j < actSz; j++) {
            pq.push({-dp[computedSources[i]][actualSources[j]], {computedSources[i], actualSources[j]}});
        }
    }

    vector <int> act, cmp;
    unordered_set <int> processed;
    int newMnSz = 0;
    for(int i = 0; i < mnSz && !pq.empty(); ) {

        auto p = pq.top();
        int dist = p.first;
        int ff = p.second.first;
        int ss = p.second.second;
        pq.pop();

        if(processed.find(ff) != processed.end() || processed.find(ss) != processed.end()) continue;

        cmp.push_back(ff);
        act.push_back(ss);
        processed.insert(ff);
        processed.insert(ss);
        i++;
        newMnSz++; 
    }

    actualSources = act;
    computedSources = cmp;
    mnSz = newMnSz;

    vector <int> candidates;

    for(int i = 0; i < mnSz; i++) {
        int ans = 0;
        int distDifference = dp[computedSources[i]][actualSources[i]];
        for(int j = 1; j <= n; j++) {
            if(dp[computedSources[i]][j] <= distDifference) ans++;
        }
        candidates.push_back(ans);
    }

    return candidates;
}

// Program to compute the distance error between the estimate source nodes and the actual source nodes
 double distError(vector <vector<pair<int, int>>> &G, int n, vector <int> &actualSources, vector <int> &computedSources, vector <vector<int>> &dp) {

    int actSz = actualSources.size();
    int cmpSz = computedSources.size();
    int mnSz = min(actSz, cmpSz);
    int mxSz = max(actSz, cmpSz);
    vector <int> distDifference(mnSz + 1);

    double distanceError = 0.00;

    priority_queue <pair<int, pair<int, int>>> pq;
    for(int i = 0; i < cmpSz; i++) {
        for(int j = 0; j < actSz; j++) {
            pq.push({-dp[computedSources[i]][actualSources[j]], {computedSources[i], actualSources[j]}});
        }
    }


    vector <int> act, cmp;
    unordered_set <int> processed;
    int newMnSz = 0; 
    for(int i = 0; !pq.empty() && i < mnSz; ) {

        auto p = pq.top();
        int dist = p.first;
        int ff = p.second.first;
        int ss = p.second.second;
        pq.pop();

        if(processed.find(ff) != processed.end() || processed.find(ss) != processed.end()) continue;

        cmp.push_back(ff);
        act.push_back(ss);
        processed.insert(ff);
        processed.insert(ss);
        i++;
        newMnSz++; 
    }

    mnSz = newMnSz; 
    actualSources = act;
    computedSources = cmp;

    cout << "Actual Sources:   ";
    for(auto x : actualSources) cout << x << " ";
    cout << endl;

    cout << "Computed Sources: ";
    for(auto y : computedSources) cout << y << " ";
    cout << endl;

    cout << "Distance Errors:  ";
    if(!mnSz) return 0; 
    int rndIdx = rand(0, mnSz - 1);
    for(int i = 0; i < mnSz; i++) {
        int dist = dp[actualSources[i]][computedSources[i]];
        if(dist == INF) dist =  vdm;
        double floatDist = dist + 0.0;
        cout << dist << " ";
        if(i == rndIdx) distanceError += floatDist;
    }
    cout << endl;

    if(!mnSz) return 0;
    distanceError /= mnSz;
    return (double)distanceError;
}

// The main() function
int32_t main() {


    double totExecTime = 0.00, totDistError = 0;
    vector <int> dde(7, 0);

    // Enter the name of the CSV file
    string fname;
    cout << "Enter the file name: ";
    cin >> fname;
    vector <int> nodes;
    vector <vector<int>> fldp;

    for(int itr = 0; itr < 50; itr++) {

        cout << "ITERATION: " << itr + 1 << endl;
        int n = 0, m =0;
        vector <vector<pair<int, int>>> G;

        // Take Graph input with edges and time delays
        inputGraph(G, n, m, nodes, fname);

        // Empty vectors to store the actual sources and the computed sources
        vector <int> actualSources, computedSources;

        // Propagate graph with the sources to get the information time of when the graph were visited first
        vector<int> infoTimes(n+7, INF);
        propogateGraph(infoTimes, G, n, actualSources, nodes);

        // Locate the sources of the graph using research paper algorithm
        clock_t start, end;
        vector <vector<int>> dp(n+7, vector<int>(n+7, INF));
        start = clock();
        locateSources(G, infoTimes, n, computedSources, dp, nodes);
        end = clock();

        // Compute the all pair shortest edge length paths
        if(!itr) {
            fldp.resize(n+7, vector<int>(n+7, INF));
            for(int i = 1; i <= n; i++) {
                for(auto x: G[i]) {
                    fldp[i][x.first] = 1;
                    fldp[x.first][i] = 1;
                }
            }

            floyd(fldp, n);
        }

        // Execution time required to locate the sources
        double executionTime = double(end - start)/ double(CLOCKS_PER_SEC);

        // The average distance error between the actual sources and computed sources
        double distanceError = distError(G, n, actualSources, computedSources, fldp);

        // The candidate source nodes based upon the distance error
        vector <int> candidates = candidateSources(G, n, actualSources, computedSources, fldp);

        cout << "Execution time: " << executionTime << endl;
        cout << "Distance Error: " << distanceError << endl;
        cout << "Candidate sources: " ;
        for(auto x : candidates) cout << x << " ";

        int integralDistance = distanceError; 
        dde[integralDistance%7]++;
        totDistError += distanceError;
        totExecTime += executionTime;
        cout << endl;

    }

    cout << "Total Distance Error: " << totDistError << endl;
    cout << "Average Distance Error: " << totDistError/50.00 << endl;
    cout << "Average Execution Time: " << totExecTime/50.00 << endl;
    cout << "Distribution of Distance Error: "; 
    for(int i = 0; i <= 6; i++) cout << i << " "; 
    cout << endl; 
    for(int i = 0; i <= 6; i++) cout << dde[i] << " ";
    cout << endl; 
    cout << "\n Press any key to continue... \n ";
    system("PAUSE");

    return 0;
}
