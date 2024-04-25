#include <bits/stdc++.h>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

void seq_lr(vector<double> &x, vector<double> &y, double &slope, double &intercept)
{
    double sum_xy = 0, sum_x = 0, sum_y = 0, sum_x_sq = 0, n = x.size();

    for (int i = 0; i < n; i++)
    {
        sum_xy += x[i] * y[i];
        sum_x += x[i];
        sum_y += y[i];
        sum_x_sq += x[i] * x[i];
    }

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x * sum_x);
    intercept = (sum_y - slope * sum_x) / n;
}

void par_lr(vector<double> &x, vector<double> &y, double &slope, double &intercept)
{
    int n = x.size();
    double sum_xy = 0, sum_x = 0, sum_y = 0, sum_x_sq = 0;

#pragma omp parallel for reduction(+ : sum_xy, sum_x, sum_y, sum_x_sq)
    for (int i = 0; i < n; i++)
    {
        sum_xy += x[i] * y[i];
        sum_x += x[i];
        sum_y += y[i];
        sum_x_sq += x[i] * x[i];
    }

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x * sum_x);
    intercept = (sum_y - slope * sum_x) / n;
}

int main()
{
    std::vector<double> x = {
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
        26.0, 27.0, 28.0, 29.0, 30.0,
        31.0, 32.0, 33.0, 34.0, 35.0,
        36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0,
        46.0, 47.0, 48.0, 49.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
        26.0, 27.0, 28.0, 29.0, 30.0,
        31.0, 32.0, 33.0, 34.0, 35.0,
        36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0,
        46.0, 47.0, 48.0, 49.0, 50.0};

    std::vector<double> y = {
        2.5, 3.5, 4.5, 5.5, 6.5,
        7.5, 8.5, 9.5, 10.5, 11.5,
        12.5, 13.5, 14.5, 15.5, 16.5,
        17.5, 18.5, 19.5, 20.5, 21.5,
        22.5, 23.5, 24.5, 25.5, 26.5,
        27.5, 28.5, 29.5, 30.5, 31.5,
        32.5, 33.5, 34.5, 35.5, 36.5,
        37.5, 38.5, 39.5, 40.5, 41.5,
        42.5, 43.5, 44.5, 45.5, 46.5,
        47.5, 48.5, 49.5, 50.5, 51.5, 2.5, 3.5, 4.5, 5.5, 6.5,
        7.5, 8.5, 9.5, 10.5, 11.5,
        12.5, 13.5, 14.5, 15.5, 16.5,
        17.5, 18.5, 19.5, 20.5, 21.5,
        22.5, 23.5, 24.5, 25.5, 26.5,
        27.5, 28.5, 29.5, 30.5, 31.5,
        32.5, 33.5, 34.5, 35.5, 36.5,
        37.5, 38.5, 39.5, 40.5, 41.5,
        42.5, 43.5, 44.5, 45.5, 46.5,
        47.5, 48.5, 49.5, 50.5, 51.5};
    double seq_slope, seq_intercept;
    double par_slope, par_intercept;

    auto start = high_resolution_clock::now();
    seq_lr(x, y, seq_slope, seq_intercept);
    auto end = high_resolution_clock::now();

    auto seq_duration = duration_cast<microseconds>(end - start);

    start = high_resolution_clock::now();
    par_lr(x, y, par_slope, par_intercept);
    end = high_resolution_clock::now();

    auto par_duration = duration_cast<microseconds>(end - start);

    cout << seq_slope << " " << seq_intercept << " " << seq_duration.count() << endl;
    cout << par_slope << " " << par_intercept << " " << par_duration.count() << endl;
}