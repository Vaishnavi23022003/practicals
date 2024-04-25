#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <omp.h>

using namespace std;

// Function to generate random data points
vector<vector<double>> generateRandomPoints(int num_points, int dim, int seed)
{
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0, 100.0);

    vector<vector<double>> points(num_points, vector<double>(dim));
    for (int i = 0; i < num_points; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            points[i][j] = distribution(generator);
        }
    }
    return points;
}

// Function to calculate Euclidean distance between two points
double euclideanDistance(const vector<double> &p1, const vector<double> &p2)
{
    double distance = 0.0;
    for (size_t i = 0; i < p1.size(); ++i)
    {
        distance += pow(p1[i] - p2[i], 2);
    }
    return sqrt(distance);
}

// Function to perform k-means clustering sequentially
vector<vector<double>> sequentialKMeans(const vector<vector<double>> &points, int k, int max_iterations)
{
    // Initialize centroids randomly
    int num_points = points.size();
    int dim = points[0].size();
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, num_points - 1);
    
    vector<vector<double>> centroids(k, vector<double>(dim));

    // choose k random points
    for (int i = 0; i < k; ++i)
    {
        int index = distribution(generator);
        centroids[i] = points[index];
    }

    // Perform k-means iterations
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // Assign each point to the nearest centroid
        vector<vector<double>> new_centroids(k, vector<double>(dim, 0.0));
        vector<int> cluster_counts(k, 0);
        for (const auto &point : points)
        {
            double min_distance = numeric_limits<double>::max();
            int closest_centroid = -1;
            for (int i = 0; i < k; ++i)
            {
                double distance = euclideanDistance(point, centroids[i]);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    closest_centroid = i;
                }
            }
            for (size_t j = 0; j < dim; ++j)
            {
                new_centroids[closest_centroid][j] += point[j];
            }
            cluster_counts[closest_centroid]++;
        }
        // Update centroids
        for (int i = 0; i < k; ++i)
        {
            if (cluster_counts[i] > 0)
            {
                for (size_t j = 0; j < dim; ++j)
                {
                    centroids[i][j] = new_centroids[i][j] / cluster_counts[i];
                }
            }
        }
    }

    return centroids;
}

// Function to perform k-means clustering using OpenMP parallelization
vector<vector<double>> parallelKMeans(const vector<vector<double>> &points, int k, int max_iterations)
{
    // Initialize centroids randomly
    int num_points = points.size();
    int dim = points[0].size();
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, num_points - 1);
    vector<vector<double>> centroids(k, vector<double>(dim));
    for (int i = 0; i < k; ++i)
    {
        int index = distribution(generator);
        centroids[i] = points[index];
    }

    // Perform k-means iterations
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // Assign each point to the nearest centroid in parallel
        vector<vector<double>> new_centroids(k, vector<double>(dim, 0.0));
        vector<int> cluster_counts(k, 0);
#pragma omp parallel for
        for (int p = 0; p < num_points; ++p)
        {
            const auto &point = points[p];
            double min_distance = numeric_limits<double>::max();
            int closest_centroid = -1;
            for (int i = 0; i < k; ++i)
            {
                double distance = euclideanDistance(point, centroids[i]);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    closest_centroid = i;
                }
            }
#pragma omp critical
            {
                for (size_t j = 0; j < dim; ++j)
                {
                    new_centroids[closest_centroid][j] += point[j];
                }
                cluster_counts[closest_centroid]++;
            }
        }
        // Update centroids
        for (int i = 0; i < k; ++i)
        {
            if (cluster_counts[i] > 0)
            {
                for (size_t j = 0; j < dim; ++j)
                {
                    centroids[i][j] = new_centroids[i][j] / cluster_counts[i];
                }
            }
        }
    }

    return centroids;
}

int main()
{
    // Parameters
    int num_points = 10000;

    // dimension of each point
    int dim = 3;

    int seed = 42;


    int k = 5;
    int max_iterations = 100;

    // Generate random data points
    vector<vector<double>> points = generateRandomPoints(num_points, dim, seed);

    // Sequential execution
    auto start_seq = chrono::steady_clock::now();
    vector<vector<double>> seq_centroids = sequentialKMeans(points, k, max_iterations);
    auto end_seq = chrono::steady_clock::now();
    cout << "Sequential Execution Time: " << chrono::duration<double, milli>(end_seq - start_seq).count() << " ms" << endl;

    // Parallel execution
    auto start_par = chrono::steady_clock::now();
    vector<vector<double>> par_centroids = parallelKMeans(points, k, max_iterations);
    auto end_par = chrono::steady_clock::now();
    cout << "Parallel Execution Time: " << chrono::duration<double, milli>(end_par - start_par).count() << " ms" << endl;

    return 0;
}

/*
Attaching output :
Sequential Execution Time: 535.433 ms
Parallel Execution Time: 439.876 ms
*/