Android
Part 5 - TD-PSOLA in Android
Now for the satisfying part. Given successive 20 ms frames from the microphone, your task will be to pitch shift to an arbitrary frequency. Because we are running in real time, the Android code will use the buffer manipulation described earlier. Epoch location detection and voiced/unvoiced detection will be implemented for you. Given a buffer setup as follows:

bufferIn:  | <-- 20 ms past --> | <-- 20 ms present --> | <-- 20 ms future --> |

bufferOut: | <-- 20 ms past --> | <-- 20 ms present --> | <-- 20 ms future --> |
COPY
you are to write the overlap-add algorithm described earlier, but with the additional constraint of the buffer manipulation algorithm. The following code is given to make implementation less tedious:

getHanningCoef(int N, int idx) returns the value of a length N Hanning window at index idx.

findClosestInVector(std::vector vector, float value, int minIdx, int maxIdx) returns the index of the value in vector which minimizes abs(vector[i] - value), minIdx inclusive and maxidx exclusive.

overlapAddArray(float *dest, float *src, int startIdx, int len) will overlap and add an array src into a larger array dest, starting at startIdx and going until startIdx + len. This function will handle array boundary checks for you.

The following variables will also be helpful:

epochLocations - A sorted vector of all epoch locations in bufferIn.

FREQ_NEW - The new desired frequency.

FRAME_SIZE - The length of a 20 ms frame.

BUFFER_SIZE - The length of the full bufferIn array.

Note
A vector in C++ is simply a variable-length array. It can be accessed just like an array, ie vector[i] = value.
Some things to keep in mind:

For a given epoch index, compute the period using p = (proceeding_epoch_idx - preceeding_epoch_idx) / 2.

When you are searching for the closest epoch using findClosestInVector(), omit the first and last entries (ie minIdx = 1, maxIdx = N-1). The period of an epoch is not well defined for the first and last epochs.

Assignment 2
Implement TD-PSOLA with buffer processing in ece420ProcessFrame(). The code to shift bufferIn and bufferOut is written for you, as well as voiced/unvoiced detection and epoch location detection.

This section will be worth 2 lab points.

Grading
Lab 5 will be graded as follows:

Prelab: 2 points

Quiz: 2 points

Lab: 4 points

Python:

Assignment 1 –> 2 points
TD-PSOLA implementation (1 point)
Various new frequency output (1 point)
Android:

Assignment 2 –> 2 point
TD-PSOLA implementation (1.5 point)
Filter functionality and app demo (0.5 point)