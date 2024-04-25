#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace std;

double euclideanDistance(const vector<double> &p1, const vector<double> &p2)
{
    double distance = 0.0;
    for (size_t i = 0; i < p1.size(); ++i)
    {
        distance += pow(p1[i] - p2[i], 2);
    }
    return sqrt(distance);
}

// Function to perform kNN sequentially
vector<int> sequentialKNN(const vector<vector<double>> &dataset, const vector<double> &query, int k)
{
    vector<pair<double, int>> distances;
    vector<int> neighbors(k);

    // Calculate distances between query point and dataset points
    for (size_t i = 0; i < dataset.size(); ++i)
    {
        double distance = euclideanDistance(query, dataset[i]);
        distances.push_back({distance, i});
    }

    // Sort distances
    sort(distances.begin(), distances.end());

    // Get the indices of k nearest neighbors
    for (int i = 0; i < k; ++i)
    {
        neighbors[i] = distances[i].second;
    }

    return neighbors;
}

// Function to perform kNN using OpenMP parallelization
vector<int> parallelKNN(const vector<vector<double>> &dataset, const vector<double> &query, int k)
{
    vector<pair<double, int>> distances;
    vector<int> neighbors(k);

// Calculate distances between query point and dataset points in parallel
#pragma omp parallel for
    for (size_t i = 0; i < dataset.size(); ++i)
    {
        double distance = euclideanDistance(query, dataset[i]);
#pragma omp critical
        distances.push_back({distance, i});
    }

    // Sort distances
    sort(distances.begin(), distances.end());

    // Get the indices of k nearest neighbors
    for (int i = 0; i < k; ++i)
    {
        neighbors[i] = distances[i].second;
    }

    return neighbors;
}

int main()
{
    // Example dataset
    vector<vector<double>> dataset = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
    vector<double> query = {3, 4, 5};
    int k = 3;

    // Sequential execution
    auto start_seq = chrono::steady_clock::now();
    vector<int> seq_neighbors = sequentialKNN(dataset, query, k);
    auto end_seq = chrono::steady_clock::now();
    for (auto i : seq_neighbors)
        cout << i << " ";
    cout << endl
         << endl;
    cout << "Sequential Execution Time: " << chrono::duration<double, milli>(end_seq - start_seq).count() << " ms" << endl;

    // Parallel execution
    auto start_par = chrono::steady_clock::now();
    vector<int> par_neighbors = parallelKNN(dataset, query, k);
    auto end_par = chrono::steady_clock::now();
    cout << "Parallel Execution Time: " << chrono::duration<double, milli>(end_par - start_par).count() << " ms" << endl;

    return 0;
}

/*
Attaching output :
Sequential Execution Time: 0.0044 ms
Parallel Execution Time: 0.5454 ms

clearly for small data overhead is too much, just in case swap the outputs if needed
*/