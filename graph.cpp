#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <string>
#include <chrono>
#include <unordered_map>
#include <omp.h>

using namespace std;
using namespace std::chrono;

class Graph {
private:
    long numNodes;
    vector<vector<long>> adjList;

public:
    Graph(long n) : numNodes(n + 1), adjList(n + 1) {}

    void addEdge(long u, long v) {
        adjList[u].push_back(v);
        adjList[v].push_back(u);
    }

    // Sequential BFS
    bool bfs(long start, long key) {
        cout << "Performing sequential BFS on the graph" << endl;
        vector<bool> visited(numNodes + 1, false);
        queue<long> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            long current = q.front();
            q.pop();

            if (current == key) {
                return true;
            }

            for (long neighbor : adjList[current]) {
                if (!visited[neighbor]) {
                    q.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }

        return false;
    }

    // Sequential DFS
    bool dfsRecursive(long current, long key, vector<bool>& visited) {
        if (current == key)
            return true;

        visited[current] = true;
        bool found = false;

        for (long neighbor : adjList[current]) {
            if (!visited[neighbor]) {
                found = dfsRecursive(neighbor, key, visited);
                if (found) break;
            }
        }

        return found;
    }

    bool dfs(long start, long key) {
        cout << "Performing sequential DFS on the graph" << endl;
        vector<bool> visited(numNodes + 1, false);
        return dfsRecursive(start, key, visited);
    }

    // Parallel BFS
    bool parallelBFS(long start, long key) {
        cout << "Performing parallel BFS on the graph" << endl;
        unordered_map<long, bool> visited;
        queue<long> q;
        q.push(start);
        visited[start] = true;
        bool found = false;

        while (!q.empty() && !found) {
#pragma omp parallel for shared(found)
            for (long i = 0; i < q.size(); ++i) {
                long current = q.front();
                q.pop();

                if (current == key) {
                    found = true; // Key found
#pragma omp cancel for
                }
                for (long j = 0; j < adjList[current].size(); ++j) {
                    long neighbor = adjList[current][j];
                    if (!visited[neighbor]) {
#pragma omp critical
                        {
                            q.push(neighbor);
                            visited[neighbor] = true;
                        }
                    }
                }
            }
        }
        return found; // Return found flag
    }

    // Parallel DFS
    bool parallelDFS(long start, long key) {
        cout << "Performing parallel DFS on the graph" << endl;
        vector<bool> visited(numNodes, false);
        return parallelDFSUtil(start, key, visited);
    }

    bool parallelDFSUtil(long current, long key, vector<bool>& visited) {
        if (current == key)
            return true;

        visited[current] = true;
        bool found = false;

#pragma omp parallel for shared(found)
        for (long neighbor : adjList[current]) {
            if (!visited[neighbor]) {
                bool localFound = parallelDFSUtil(neighbor, key, visited);
                if (localFound) {
#pragma omp atomic write
                    found = true; // Update found flag atomically
                }
            }
        }
        return found;
    }
};

int main() {
    long vertices, edges;
    string trash;
    char choose;



    cout << "Enter the number of vertices: ";
    cin >> vertices;

    Graph graph(vertices);

    cout << "Enter the number of edges: ";
    cin >> edges;

    long v1, v2;
    cout << "Enter the edges (format: vertex1 vertex2):" << endl;
    for (long i = 0; i < edges; i++) {
        cin >> v1 >> v2;
        graph.addEdge(v1, v2);
    }
    do{
    long key;
    cout << "Enter the key to search: ";
    cin >> key;
    omp_set_num_threads(16);
    int choice;
    cout << "Choose the algorithm to search for key " << key << ":" << endl;
    cout << "1. BFS" << endl;
    cout << "2. DFS" << endl;
    cout << "Enter your choice: ";
    cin >> choice;

    auto start = high_resolution_clock::now();
    bool found;

    if (choice == 1) {
        found = graph.bfs(1, key);
    } else if (choice == 2) {
        found = graph.dfs(1, key);
    } else {
        cout << "Invalid choice. Exiting." << endl;
        return 1;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);

    if (found) {
        cout << "Key " << key << " found." << endl;
    } else {
        cout << "Key " << key << " not found." << endl;
    }

    cout << "Time taken for sequential algorithm: " << duration.count() << " nanoseconds" << endl;

    // Now, we compare with the parallel version
    auto start_parallel = high_resolution_clock::now();
    bool found_parallel;

    if (choice == 1) {
        found_parallel = graph.parallelBFS(1, key);
    } else if (choice == 2) {
        found_parallel = graph.parallelDFS(1, key);
    }

    auto end_parallel = high_resolution_clock::now();
    auto duration_parallel = duration_cast<nanoseconds>(end_parallel - start_parallel);

    if (found_parallel) {
        cout << "Key " << key << " found in parallel version." << endl;
    } else {
        cout << "Key " << key << " not found in parallel version." << endl;
    }

    cout << "Time taken for parallel algorithm: " << duration_parallel.count() << " nanoseconds" << endl;


    cout<<"\n continue(y/n): ";
    cin>>choose;

    }while(tolower(choose) == 'y');

    return 0;
}
