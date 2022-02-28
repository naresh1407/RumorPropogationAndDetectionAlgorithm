#include <bits/stdc++.h> 
using namespace std; 

#define INF (int)(1e9 + 7)

void inputGraph(vector <vector<pair<int, int>>> &G, int m) {
    int from, to, timeDelay; 
    for(int i = 1; i <= m; i++) {
        cin >> from >> to >> timeDelay; 
        G[from].push_back({to, timeDelay}); 
        G[to].push_back({from, timeDelay}); 
    }
}

void propDijkstra(vector<vector<pair<int,int>>> &G, int source, int n, 
    vector <int> &infoTimes, int timeInit) {

    vector <bool> visited(n+1, false); 
    infoTimes[source] = timeInit; 

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
                infoTimes[adjacentNode] = infoTimes[chosenNode] + delay; 
            }
        }
    }

}

void propogateGraph(vector <int> &infoTimes, vector <vector<pair<int, int>>> &G, int n) {

    int k, timeInit; 
    cin >> k >> timeInit; 
    
    vector <int> sources;  
    for(int i = 0 ; i < k; i++) {
        int currSource; 
        cin >> currSource; 
        sources.push_back(currSource); 
        infoTimes[currSource] = timeInit; 
    }

    for(auto currSource : sources) {
        propDijkstra(G, currSource, n, infoTimes, timeInit); 
    }
    cout << "Nodes in graph: "; 
    for(int i = 1; i <= n; i++) cout << char('a' + i - 1) << " "; 
    cout << endl; 
    cout << "Informed times: ";
    for(int i = 1; i <= n; i++) cout << infoTimes[i] << " "; 
    cout << endl; 
}

void locDijkstra(vector <vector<int>> &minDelay, vector <vector<pair<int, int>>> &G, vector <int> &infoTimes, int n, int observer) {

    vector <bool> visited(n+1, false); 
    minDelay[observer][observer] = infoTimes[observer];

    for(int iteration = 0; iteration < n; iteration++) {
        
        int chosenNode = -1; 
        for(int currNode = 1; currNode <= n; currNode++) {
            if(!visited[currNode] && (chosenNode == -1 || minDelay[observer][currNode] < minDelay[observer][chosenNode])) {
                chosenNode = currNode; 
            }
        }

        if(minDelay[observer][chosenNode] == INF) break; 

        visited[chosenNode] = true; 

        for(auto edge : G[chosenNode]) {

            int adjacentNode = edge.first; 
            int delay = edge.second; 
            
            if(minDelay[observer][adjacentNode] == INF && (minDelay[observer][chosenNode] - delay < minDelay[observer][adjacentNode]))
                minDelay[observer][adjacentNode] = minDelay[observer][chosenNode] - delay;
        }
    }

}


void locateSources(vector <vector<pair<int, int>>> &G, vector <int> &informedTimes, int n) {
    
    int observerCount; 
    cin >> observerCount; 
    vector <int> observers; 
    vector <vector<int>> minDelay(n + 1, vector<int>(n+1, INF));
    
    int minObservationTime = INF; 
    for(int i = 0; i < observerCount; i++) {
        int observer; 
        cin >> observer; 
        observers.push_back(observer); 
        minDelay[observer][observer] = informedTimes[observer]; 
        minObservationTime = min(minObservationTime, informedTimes[observer]); 
    } 
    
    for(auto observer : observers) {
        locDijkstra(minDelay, G, informedTimes, n, observer);
    }
    
    cout << "Observer \t"; 
    for(int i = 1; i <= n; i++) cout << char('a' + i - 1) << "\t"; 
    cout << endl; 
    cout << endl; 
    
    for(auto observer: observers) {
        cout << char(observer + 'a' - 1) << "\t \t"; 
        for(int i = 1;  i <= n; i++) cout << minDelay[observer][i] << "\t"; 
        cout << endl; 
    }

    vector <pair<int, int>> maxDistAndFreq(n+1, {0, 0}); 

    for(auto observer : observers) {
        for(int i = 1; i <= n; i++) {
            maxDistAndFreq[i].first = max(maxDistAndFreq[i].first, minDelay[observer][i]); 
        }
    }

    int maxFreq = 0;
    for(auto observer : observers) {
        for(int i = 1; i <= n; i++) {
            if(minDelay[observer][i] == maxDistAndFreq[i].first) {
                maxDistAndFreq[i].second++; 
                maxFreq = max(maxFreq, maxDistAndFreq[i].second); 
            }
        }
    }

    vector <int> sources; 
    int timeInit = INF; 

    for(int i = 1; i <= n; i++) {

        int relativeTime = maxDistAndFreq[i].first; 
        int freq = maxDistAndFreq[i].second; 

        if(relativeTime <= minObservationTime && freq == maxFreq) {
            sources.push_back(i); 
            timeInit = relativeTime; 
        }
    }

    cout << "Sources: "; 
    for(auto x: sources) cout << char(x + 'a' - 1) << " "; 
    cout << endl; 

    cout  << "Initial time: " << timeInit << endl; 
}

int main() {

    
    // Take Graph input with edges and time delays
    int n, m; 
    cin >> n >> m; 
    vector <vector<pair<int, int>>> G(n+1);  
    inputGraph(G, m); 
    
    
    // Propagate graph with the sources to get the information time of when the graph were visited first
    vector<int> infoTimes(n+1, INF);
    propogateGraph(infoTimes, G, n);

    // Locate the sources of the graph using research paper algorithm
    locateSources(G, infoTimes, n); 

    return 0; 
}