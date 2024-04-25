#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <fstream>
const int SEQUENTIAL_CUTOFF = 1000; // Adjust this threshold as needed

using namespace std;

// Function prototypes
void bubbleSort(vector<long>& arr, long n);
void parallelBubbleSort(vector<long>& array, long n);
void mergeSort(vector<long>& arr, long l, long r);
void parallelMergeSort(vector<long>& arr, long l, long r);
void merge(vector<long>& arr, long l, long m, long r);
void displayArray(const vector<long>& arr);
void displayComparisonTable(double seqBubbleTime, double parBubbleTime, double seqMergeTime, double parMergeTime);

int main() {
    char choice;
    do {
        long sortingChoice;
        cout << "Menu:\n";
        cout << "1. Bubble Sort\n";
        cout << "2. Merge Sort\n";
        cout << "Enter your choice (1 or 2): ";
        cin >> sortingChoice;

        if (sortingChoice != 1 && sortingChoice != 2) {
            cout << "Invalid choice. Exiting program.\n";
            break; // Exit the loop if the choice is invalid
        }

        vector<long> nums;
        long rangeStart, rangeEnd;

        cout << "Enter the range of numbers (start end): ";
        cin >> rangeStart >> rangeEnd;

        srand(time(0)); // Seed for random number generation

        // Generate random numbers within the specified range
        for (long i = rangeStart; i <= rangeEnd; ++i) {
            nums.push_back(rand() % (rangeEnd - rangeStart + 1) + rangeStart);
        }
        // long thread_nums;
        // cout<<"\nEnter number of threads use:  ";
        // cin>> thread_nums;

        // omp_set_num_threads(thread_nums);

        if (sortingChoice == 1) {
            vector<long> copy_nums = nums;

            // Sequential Bubble Sort
            long n = nums.size();
            auto startTimeSeqBubble = chrono::high_resolution_clock::now();
            bubbleSort(nums, n);
            auto endTimeSeqBubble = chrono::high_resolution_clock::now();
            double seqBubbleTime = chrono::duration<double>(endTimeSeqBubble - startTimeSeqBubble).count();

            // Parallel Bubble Sort
            auto startTimeParBubble = chrono::high_resolution_clock::now();
            parallelBubbleSort(copy_nums, n);
            auto endTimeParBubble = chrono::high_resolution_clock::now();
            double parBubbleTime = chrono::duration<double>(endTimeParBubble - startTimeParBubble).count();

            // Display comparison table for Bubble Sort
            displayComparisonTable(seqBubbleTime, parBubbleTime, 0.0, 0.0);

            // cout << endl << "The sorted array-->\n";
            // for (auto x : nums) {
            //     cout << x << " ";
            // }
        } else if (sortingChoice == 2) {
            vector<long> merge_nums = nums;
            vector<long> parallel_merge_nums = nums;

            // Sequential Merge Sort
            auto startTimeSeqMerge = chrono::high_resolution_clock::now();
            mergeSort(merge_nums, 0, merge_nums.size() - 1);
            auto endTimeSeqMerge = chrono::high_resolution_clock::now();
            double seqMergeTime = chrono::duration<double>(endTimeSeqMerge - startTimeSeqMerge).count();

            // Parallel Merge Sort
            auto startTimeParMerge = chrono::high_resolution_clock::now();
            parallelMergeSort(parallel_merge_nums, 0, parallel_merge_nums.size() - 1);
            auto endTimeParMerge = chrono::high_resolution_clock::now();
            double parMergeTime = chrono::duration<double>(endTimeParMerge - startTimeParMerge).count();

            // Display comparison table for Merge Sort
            displayComparisonTable(0.0, 0.0, seqMergeTime, parMergeTime);

            // cout << endl << "The sorted array-->\n";
            // for (auto x : parallel_merge_nums) {
            //     cout << x << " ";
            // }
        }

        cout << "\n\nDo you want to continue (y/n)? ";
        cin >> choice;

    } while (tolower(choice) == 'y');

    return 0;
}

void bubbleSort(vector<long>& arr, long n) {
    for (long i = 0; i < n - 1; ++i) {
        for (long j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1])  swap(arr[j], arr[j + 1]);
        }
    }
}

void parallelBubbleSort(vector<long>& array, long n) {
    for (long i = 0; i < n; ++i) {
        #pragma omp parallel for shared(array)num_threads(16)
        for (long j = 1; j < n; j += 2) {
            if (array[j] < array[j - 1]) {
                swap(array[j], array[j - 1]);
            }
        }
        #pragma omp barrier
        #pragma omp parallel for shared(array)num_threads(16)
        for (long j = 2; j < n; j += 2) {
            if (array[j] < array[j - 1]) {
                swap(array[j], array[j - 1]);
            }
        }
    }
}


void merge(std::vector<long>& arr, long left, long middle, long right) {
    long n1 = middle - left + 1;
    long n2 = right - middle;

    // Create temporary arrays
    vector<long> leftArray(n1);
    vector<long> rightArray(n2);

    // Copy data to temporary arrays
    for (long i = 0; i < n1; ++i)
        leftArray[i] = arr[left + i];
    for (long j = 0; j < n2; ++j)
        rightArray[j] = arr[middle + 1 + j];

    // Merge the temporary arrays back into arr
    long i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (leftArray[i] <= rightArray[j])
            arr[k++] = leftArray[i++];
        else
            arr[k++] = rightArray[j++];
    }

    // Copy remaining elements of leftArray
    while (i < n1)
        arr[k++] = leftArray[i++];

    // Copy remaining elements of rightArray
    while (j < n2)
        arr[k++] = rightArray[j++];
}

// Sequential Merge Sort
void mergeSort(std::vector<long>& arr, long left, long right) {
    if (left >= right) return;      // the base condition

    long middle = left + (right - left) / 2;     // dividing array into two parts
    mergeSort(arr, left, middle);
    mergeSort(arr, middle + 1, right);
    merge(arr, left, middle, right);        // merging  both sorted parts
}

void parallelMergeSort(vector<long>& arr, long left, long right) {
    if (left >= right) return;

    long middle = left + (right - left) / 2;

    if (right - left <= SEQUENTIAL_CUTOFF) {
        mergeSort(arr, left, right);
    } else {
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                parallelMergeSort(arr, left, middle);

                #pragma omp task
                parallelMergeSort(arr, middle + 1, right);
            }
        }
    }

    merge(arr, left, middle, right);
}


void displayComparisonTable(double seqBubbleTime, double parBubbleTime, double seqMergeTime, double parMergeTime) {
    // Display comparison table
    cout << "Algorithm        | Sequential Time (s) | Parallel Time (s)" << endl;
    cout << "---------------------------------------------------------" << endl;
    cout << "Bubble Sort      | " << seqBubbleTime << "                 | " << parBubbleTime << endl;
    cout << "Merge Sort       | " << seqMergeTime << "                 | " << parMergeTime << endl;
}
