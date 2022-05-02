#include <bits/stdc++.h>
#include <sstream>
using namespace std;

#define INF (int)(1e9 + 7)

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
int rand(int l, int r){
  uniform_int_distribution<int> ludo(l, r); return ludo(rng);
}

int mystoi(string s) {
    int res = 0; 
    int n = s.length(); 
    for(int i = 0; i < n; i++) {
        int currChar = s[i] - '0';
        res = res * 10LL + currChar; 
    }
    return res; 
}

void inputGraph(vector<vector<pair<int, int>>> &G, int &n, int &m) {

    int from, to, timeDelay;
    string fname;
    cout << "Enter the file name: ";
    cin >> fname;

    cout << "\n \nProcessing the graph ... \n \n";

    //Copy the contents of the csv file into the vector 'content'
    vector<vector<string>> content;
    vector<string> row;
    string line, word;

    fstream file(fname, ios::in);

    if (file.is_open()) {

        while (getline(file, line)) {
            row.clear();
            stringstream str(line);
            while (getline(str, word, ',')) row.push_back(word);
            content.push_back(row);
            }
    }
    else{
        cout << "Could not open the file\n";
        return ; 
    }

    file.close(); 


    //Get no. of edges
    m = content.size();
    G.resize(5000);

     
    //Create the graph
    for (int i = 0; i < m; i++) {
        from = mystoi(content[i][0]);
        to = mystoi(content[i][1]);
        timeDelay = rand(1LL, 10LL);
        
        // cout<<from<<" "<<to<<" "<<timeDelay<<"\n";
        G[from].push_back({to, timeDelay});
        G[to].push_back({from, timeDelay});
        // cout << m << endl;
        
    }

    cout<<"\n";

    cout << "Graph processed.\n \n";
}

void propDijkstra(vector<vector<pair<int,int>>> &G, int source, int n,
    vector <int> &infoTimes, int timeInit) {

    vector <bool> visited(n+1, false);
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

void propogateGraph(vector <int> &infoTimes, vector <vector<pair<int, int>>> &G, int n, vector <int> &actualSources) {

    int k = n/10, timeInit = rand(1000, 5000);

    cout << "Initial diffusion time: " << timeInit << endl;
    vector <int> sources;
    cout << "Actual diffusion sources: ";

    unordered_set <int> processed;
    for(int i = 0 ; i < k; i++) {
        int currSource;
        while(1) {
            currSource = rand(1, n);
            if(processed.find(currSource) != processed.end()) continue;
            cout << currSource << " ";
            processed.insert(currSource);
            sources.push_back(currSource);
            break;
        }
        infoTimes[currSource] = timeInit;
    }

    cout << endl;
    actualSources = sources;


    cout << "\n Diffusing the information in the graph... \n \n";
    for(auto currSource : sources) {
        propDijkstra(G, currSource, n, infoTimes, timeInit);
    }
    cout << "\n Information diffused in the graph. \n";

    cout << "\n Informed times of Nodes in the graph: \n";
    for(int i = 1; i <= n; i++) cout << infoTimes[i] << " ";
    cout << endl;

}

void floyd(vector <vector<int>> &dp, int n) {
    for(int k = 1; k <= n; k++) {
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= n; j++) {
                if(dp[i][k] < INF && dp[k][j] < INF)
                dp[i][j] = min(dp[i][j], (dp[i][k] + dp[k][j]));
            }
        }
    }
}


void locateSources(vector <vector<pair<int, int>>> &G, vector <int> &informedTimes, int n, vector <int> &calcluatedSources,
    vector <vector<int>> &dp) {

    int observerCount = n/10;
    vector <int> observers;
    vector <vector<int>> minDelay(n + 7, vector<int>(n+7, INF));

    int minObservationTime = INF;
    unordered_set <int> processed;

    cout << "Number of nodes computed: " << n << endl; 
    cout << endl; 
    

    cout << "Observers: ";
    for(int i = 0; i < observerCount; i++) {
        int observer;
        while(1) {
            observer = rand(1, n);
            if(processed.find(observer) != processed.end()) continue;
            processed.insert(observer);
            // cout << observer << " ";
            observers.push_back(observer);
            break ;
        }
        minDelay[observer][observer] = informedTimes[observer];
        minObservationTime = min(minObservationTime, informedTimes[observer]);
    }
    cout << endl;
    cout << endl;


    for(int i = 1; i <= n; i++) {
        for(auto x: G[i]) {
            dp[i][x.first] = x.second;
            dp[x.first][i] = x.second;
        }
    }

    cout << "\n Locating the sources... \n \n";
    cout << "\n Computing distances... \n \n";
    floyd(dp, n);
    cout << "\n Distances computed... \n \n";

    for(auto observer : observers) {
        for(int i = 1; i <= n; i++) {
            minDelay[observer][i] = minDelay[observer][observer] - dp[observer][i];
        }
    }

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

    calcluatedSources = sources;

    cout << "\n Sources computed ...\n \n";
    cout  << "Initial time: " << timeInit << endl;
}

int searchCandidates(vector <bool> &visited, vector <vector<pair<int, int>>> &G, int source, int dist) {

    int candidateCount = 0;
    queue <pair<int, int>> q;
    q.push({source, 0});
    visited[source] = 1;

    while(!q.empty()) {

        int sz = q.size();
        for(int i = 0; i < sz; i++) {
            auto front = q.front();
            q.pop();
            visited[front.first] = 1;
            if(front.second <= dist) candidateCount++;
            for(auto x: G[front.first]) {
                if(!visited[x.first]) q.push({x.first, front.second + 1});
            }
        }
    }

    return candidateCount;
}

vector <int> candidateSources(vector <vector<pair<int, int>>> &G, int n, vector <int> &actualSources, vector <int> &computedSources) {

    int actSz = actualSources.size();
    int cmpSz = computedSources.size();
    int mnSz = min(actSz, cmpSz);
    int mxSz = max(actSz, cmpSz);

    vector <vector<int>> dp(n+7, vector<int>(n+7, INF));
    for(int i = 1; i <= n; i++) {
        for(auto x: G[i]) {
            dp[i][x.first] = 1;
            dp[x.first][i] = 1;
        }
    }

    floyd(dp, n);

    priority_queue <pair<int, pair<int, int>>> pq;
    for(int i = 0; i < cmpSz; i++) {
        for(int j = 0; j < actSz; j++) {
            pq.push({-dp[computedSources[i]][actualSources[j]], {computedSources[i], actualSources[j]}});
        }
    }

    vector <int> act, cmp;
    unordered_set <int> processed;
    for(int i = 0; i < mnSz; ) {

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
    }

    actualSources = act;
    computedSources = cmp;


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

 double distError(vector <vector<pair<int, int>>> &G, int n, vector <int> &actualSources, vector <int> &computedSources) {

    int actSz = actualSources.size();
    int cmpSz = computedSources.size();
    int mnSz = min(actSz, cmpSz);
    int mxSz = max(actSz, cmpSz);
    vector <int> distDifference(mnSz + 7);

    double distanceError = 0.00;

    vector <vector<int>> dp(n+7, vector<int>(n+7, INF));
    for(int i = 1; i <= n; i++) {
        for(auto x: G[i]) {
            dp[i][x.first] = 1;
            dp[x.first][i] = 1;
        }
    }

    floyd(dp, n);

    priority_queue <pair<int, pair<int, int>>> pq;
    for(int i = 0; i < cmpSz; i++) {
        for(int j = 0; j < actSz; j++) {
            pq.push({-dp[computedSources[i]][actualSources[j]], {computedSources[i], actualSources[j]}});
        }
    }


    vector <int> act, cmp;
    unordered_set <int> processed;
    for(int i = 0; i < mnSz; ) {

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
    }

    actualSources = act;
    computedSources = cmp;

    cout << "Actual Sources:   ";
    for(auto x : actualSources) cout << x << " ";
    cout << endl;

    cout << "Computed Sources: ";
    for(auto y : computedSources) cout << y << " ";
    cout << endl;

    cout << "Distance Errors:  ";
    for(int i = 0; i < mnSz; i++) {
        int dist = dp[actualSources[i]][computedSources[i]];
        double floatDist = dist + 0.0;
        cout << dist << " ";
        distanceError += floatDist;
    }
    cout << endl;

    if(!mnSz) return 0;
    distanceError /= mnSz;
    return (double)distanceError;
}


int32_t main() {

    // ios_base:: sync_with_stdio(false);
	// cin.tie(0);
	// cout.tie(0);

    clock_t start, end;
    start = clock();

    // Take Graph input with edges and time delays
    int n = 0, m =0;
    vector <vector<pair<int, int>>> G;
    inputGraph(G, n, m);

    // Empty vectors to store the actual sources and the computed sources
    vector <int> actualSources, computedSources;

    // Propagate graph with the sources to get the information time of when the graph were visited first
    vector<int> infoTimes(n+7, INF);
    propogateGraph(infoTimes, G, n, actualSources);


    // Locate the sources of the graph using research paper algorithm
    vector <vector<int>> dp(n+7, vector<int>(n+7, INF));
    locateSources(G, infoTimes, n, computedSources, dp);
    

    
    // The average distance error between the actual sources and computed sources
    double distanceError = distError(G, n, actualSources, computedSources);

    // The candidate source nodes based upon the distance error
    vector <int> candidates = candidateSources(G, n, actualSources, computedSources);

    cout << "Distance Error: " << (double)distanceError << endl;
    cout << "Candidate sources: " ;
    for(auto x : candidates) cout << x << " ";
    cout << endl;
    end = clock();
    // Execution time required to locate the sources
    double executionTime = double(end - start)/ double(CLOCKS_PER_SEC);

    cout << "Execution time: " << executionTime << endl;
    

    return 0;
}
