#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Function prototypes
long minvalSequential(const vector<long>& nums, long n);
long maxvalSequential(const vector<long>& nums, long n);
long sumSequential(const vector<long>& nums, long n);
double averageSequential(const vector<long>& nums, long n);
long minvalParallel(const vector<long>& nums, long n);
long maxvalParallel(const vector<long>& nums, long n);
long sumParallel(const vector<long>& nums, long n);
double averageParallel(const vector<long>& nums, long n);
void displayComparisonTable(const std::chrono::duration<long, std::nano>& seqTime, const std::chrono::duration<long, std::nano>& parTime, const std::string& operation);


int main() {
    long choice;
    cout << "Menu:\n";
    cout << "1. Generate Random Numbers\n";
    cout << "2. Input Numbers from User\n";
    cout << "Enter your choice: ";
    cin >> choice;

    vector<long> nums;
    long rangeStart, rangeEnd;

    if (choice == 1) {
        cout << "Enter the range of numbers (start end): ";
        cin >> rangeStart >> rangeEnd;
        srand(time(0));
        for (long i = rangeStart; i <= rangeEnd; ++i) {
            nums.push_back(rand() % (rangeEnd - rangeStart + 1) + rangeStart);
        }
    } else if (choice == 2) {
        long numElements;
        cout << "Enter the number of elements: ";
        cin >> numElements;
        cout << "Enter " << numElements << " numbers:\n";
        for (long i = 0; i < numElements; ++i) {
            long num;
            cin >> num;
            nums.push_back(num);
        }
    } else {
        cout << "Invalid choice. Exiting program.\n";
        return 0;
    }

    // Sequential Execution
    auto startSeq = high_resolution_clock::now();
    long minSeq = minvalSequential(nums, nums.size());
    auto stopSeq = high_resolution_clock::now();
    auto durationSeq = duration_cast<nanoseconds>(stopSeq - startSeq);

    auto startSeq2 = high_resolution_clock::now();
    long maxSeq = maxvalSequential(nums, nums.size());
    auto stopSeq2 = high_resolution_clock::now();
    auto durationSeq2 = duration_cast<nanoseconds>(stopSeq2 - startSeq2);

    auto startSeq3 = high_resolution_clock::now();
    long sumSeq = sumSequential(nums, nums.size());
    auto stopSeq3 = high_resolution_clock::now();
    auto durationSeq3 = duration_cast<nanoseconds>(stopSeq3 - startSeq3);

    auto startSeq4 = high_resolution_clock::now();
    double avgSeq = averageSequential(nums, nums.size());
    auto stopSeq4 = high_resolution_clock::now();
    auto durationSeq4 = duration_cast<nanoseconds>(stopSeq4 - startSeq4);

    // Parallel Execution
    auto startPar = high_resolution_clock::now();
    long minPar = minvalParallel(nums, nums.size());
    auto stopPar = high_resolution_clock::now();
    auto durationPar = duration_cast<nanoseconds>(stopPar - startPar);

    auto startPar2 = high_resolution_clock::now();
    long maxPar = maxvalParallel(nums, nums.size());
    auto stopPar2 = high_resolution_clock::now();
    auto durationPar2 = duration_cast<nanoseconds>(stopPar2 - startPar2);

    auto startPar3 = high_resolution_clock::now();
    long sumPar = sumParallel(nums, nums.size());
    auto stopPar3 = high_resolution_clock::now();
    auto durationPar3 = duration_cast<nanoseconds>(stopPar3 - startPar3);

    auto startPar4 = high_resolution_clock::now();
    double avgPar = averageParallel(nums, nums.size());
    auto stopPar4 = high_resolution_clock::now();
    auto durationPar4 = duration_cast<nanoseconds>(stopPar4 - startPar4);

    // Display comparison tables
    displayComparisonTable(durationSeq, durationPar, "Minimum");
    displayComparisonTable(durationSeq2, durationPar2, "Maximum");
    displayComparisonTable(durationSeq3, durationPar3, "Sum");
    displayComparisonTable(durationSeq4, durationPar4, "Average");

    return 0;
}

// Function definitions...

void displayComparisonTable(const duration<long, std::nano>& seqTime, const duration<long, std::nano>& parTime, const string& operation) {
    cout << "Comparison Table for " << operation << " Time:\n";
    cout << "---------------------------------------------------------\n";
    cout << "| Metric       | Sequential (ms) | Parallel (ms) |\n";
    cout << "---------------------------------------------------------\n";
    cout << "| Time         | " << seqTime.count() << "              | " << parTime.count() << "             |\n";
    cout << "---------------------------------------------------------\n";
}


void generateRandomNumbers(vector<long>& nums, long rangeStart, long rangeEnd) {
    srand(time(0));
    for (long i = rangeStart; i <= rangeEnd; ++i) {
        nums.push_back(rand() % (rangeEnd - rangeStart + 1) + rangeStart);
    }
}

long minvalSequential(const vector<long>& nums, long n) {
    long minval = nums[0];
    for (long i = 0; i < n; i++) {
        if (nums[i] < minval) minval = nums[i];
    }
    return minval;
}

long maxvalSequential(const vector<long>& nums, long n) {
    long maxval = nums[0];
    for (long i = 0; i < n; i++) {
        if (nums[i] > maxval) maxval = nums[i];
    }
    return maxval;
}

long sumSequential(const vector<long>& nums, long n) {
    long sum = 0;
    for (long i = 0; i < n; i++) {
        sum += nums[i];
    }
    return sum;
}

double averageSequential(const vector<long>& nums, long n) {
    return static_cast<double>(sumSequential(nums, n)) / n;
}

long minvalParallel(const vector<long>& nums, long n) {
    long minval = nums[0];
    #pragma omp parallel for reduction(min:minval)
    for (long i = 0; i < n; i++) {
        if (nums[i] < minval) minval = nums[i];
    }
    return minval;
}

long maxvalParallel(const vector<long>& nums, long n) {
    long maxval = nums[0];
    #pragma omp parallel for reduction(max:maxval)
    for (long i = 0; i < n; i++) {
        if (nums[i] > maxval) maxval = nums[i];
    }
    return maxval;
}

long sumParallel(const vector<long>& nums, long n) {
    long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; i++) {
        sum += nums[i];
    }
    return sum;
}

double averageParallel(const vector<long>& nums, long n) {
    return static_cast<double>(sumParallel(nums, n)) / n;
}
