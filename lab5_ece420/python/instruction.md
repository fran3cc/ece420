
Lab 5
Lab 5 - Pitch Synthesis
Summary
In this lab, you will learn about a simplified model for human speech and how to exploit this model to shift the frequency of an incoming signal. You will also learn one method for producing continuous output within a batch processing framework.

Downloads
test_vector_all_voiced_epochs.npy

test_vector_all_voiced.wav

Android project source code

Android project source code with spectrogram (Use this optional Android project if you want to test the TD_PSOLA output by the spectrogram.)

Note
This video on doing TD-PSOLA by hand is very helpful for getting an intuitive understanding of TD-PSOLA.

We have completed the scaffolding of the TD-PSOLA algorithm for you, leaving remapping, windowing, and overlap-add for your lab assignment. You will be responsible for knowing why our buffering scheme guarantees smoothness in the signal reconstruction, and you will also be responsible for knowing why the characteristics of human speech permit such an algorithm.

We will try to explain these in detail in the lab notes, but please do not hesitate to ask your TA or ask on Campuswire if these details are unclear.

TD-PSOLA
Time-Domain Pitch Synchronous Overlap-Add, or TD-PSOLA, is the pitch-shifting algorithm we will be using in this lab. It relies heavily the source-filter model of speech, detailed below.

Part 1 - Source-Filter Model of Speech
We can model speech as the output of two successive blocks, a source block and filter block:



Our vocal cords are the source, and we model the output of our vocal cords as a delta train (also called a Dirac comb). Of course, our vocal cords cannot truly produce zero-width impulses of energy, but this assumption is sufficient for modeling purposes. This series of impulses determines the fundamental frequency, or the pitch, of our utterance.

The impulses of energy produced by our vocal cords then travel up through our vocal tract. Everything in your vocal tract (lips, tongue throat, mouth cavity, nasal cavity, among others) affect the final waveform. We make the assumption that these effects are linear and can be combined without penalty, so we call everything inside the vocal tract the filter.

Simplifying the complex problem of speech generation in this manner allows us to apply some signal processing tools we learned in ECE 310. In particular, if we assume that speech is nothing but the output of a filter, we can rewrite speech in the form of a transfer function where 
�
(
�
)
y(t) is the utterance, 
�
(
�
)
x(t) is the source, and 
ℎ
(
�
)
h(t) is the filter:

�
(
�
)
=
�
(
�
)
∗
ℎ
(
�
)
y(t)=x(t)∗h(t)
Note that 
∗
∗ denotes the convolution operator. We can further rewrite 
�
(
�
)
x(t), assuming 
�
(
�
)
x(t) is a delta train with some period P:

�
(
�
)
=
(
∑
�
=
0
∞
�
(
�
−
�
�
)
)
∗
ℎ
(
�
)
=
∑
�
=
0
∞
�
(
�
−
�
�
)
∗
ℎ
(
�
)
=
∑
�
=
0
∞
ℎ
(
�
−
�
�
)
y(t)
​
  
=( 
k=0
∑
∞
​
 δ(t−kP))∗h(t)
= 
k=0
∑
∞
​
 δ(t−kP)∗h(t)
= 
k=0
∑
∞
​
 h(t−kP)
​
 
This is a powerful result which says that the fundamental frequency (the pitch) does not depend on the filter 
ℎ
(
�
)
h(t), but on the source 
�
(
�
)
x(t). In other words, speech can be decomposed into a sum of impulse responses, where the spacing of the impulse responses P determines the pitch.

If we want to shift the pitch, all we need is some way of identifying the impulse response from the speech signal. We can then extract the impulse response and place it with any spacing P in the synthesis signal we desire.

Part 2 - TD-PSOLA Explained
Considering the discussion above, we can rewrite our problem more formally. Given a signal 
�
�
0
(
�
)
y 
P 
0
​
 
​
 (t) with some fundamental period 
�
0
P 
0
​
 :

�
�
0
(
�
)
=
∑
�
=
0
∞
ℎ
(
�
−
�
�
0
)
,
y 
P 
0
​
 
​
 (t)= 
k=0
∑
∞
​
 h(t−kP 
0
​
 ),
compute a new synthesis signal y{P1}(t) with a new period P_1:

�
�
1
(
�
)
=
∑
�
=
0
∞
ℎ
(
�
−
�
�
1
)
.
y 
P 
1
​
 
​
 (t)= 
k=0
∑
∞
​
 h(t−kP 
1
​
 ).
In order to compute the new synthesis signal, we need some way of extracting the impulse response from 
�
�
0
(
�
)
y 
P 
0
​
 
​
 (t). This is done by identifying the period (using the technique from Lab 4) and further identifying each epoch in the signal, or the location of each pitch period, by finding peaks in the signal approximately 
�
0
P 
0
​
  samples apart. An example of this for a sample frame is shown below.



We can then extract the impulse response (rather, an estimate of the impulse response) by windowing 
±
�
0
±P 
0
​
  about each epoch marker. For example, when windowed with a Hanning window, an individual response may look like this:



Now that we have a method of estimating the impulse responses, we simply need to map from the original spacing 
�
�
0
(
�
)
=
∑
�
=
0
∞
ℎ
(
�
−
�
�
0
)
y 
P 
0
​
 
​
 (t)=∑ 
k=0
∞
​
 h(t−kP 
0
​
 ) to the synthesis spacing 
�
�
1
(
�
)
=
∑
�
=
0
∞
ℎ
(
�
−
�
�
1
)
y 
P 
1
​
 
​
 (t)=∑ 
k=0
∞
​
 h(t−kP 
1
​
 ). In other words, we need to determine where the original impulse responses should go in our new signal.

Imagine our original signal has epochs spaced as below:

orig: 1     2     3     4     5     6     7     8
COPY
and we want to double the frequency of the original signal without changing the duration. This can be achieved by spacing our new epochs at 
�
1
=
�
0
2
P 
1
​
 = 
2
P 
0
​
 
​
 . For every new epoch at 
{
�
1
,
2
�
1
,
3
�
1
,
…
 
}
{P 
1
​
 ,2P 
1
​
 ,3P 
1
​
 ,…}, we need to find the nearest epoch in the original signal. This mapping is shown below for our example signal:

orig: 1     2     3     4     5     6     7     8
      |\    |\    |\    |\    |\    |\    |\    |
      | \   | \   | \   | \   | \   | \   | \   |
new:  1  1  2  2  3  3  4  4  5  5  6  6  7  7  8
COPY
The impulse responses in the new signal are combined by overlapping and adding their components. In Python, overlap and add is simply:

orig_signal = np.zeros(N)

# Window, compute impulse response
...

for new_epoch_idx in range(0, N, P_1):
    orig_signal[new_epoch_idx - P_0:new_epoch_idx + P_0 + 1] += windowed_response
COPY
The code above does not check if array indices are valid, so be careful.

Likewise, consider halving the frequency. This can be achieved with the following mapping:

orig: 1     2     3     4     5     6     7     8
      |           |           |           |
      |           |           |           |
new:  1           3           5           7
COPY
Overlap-Add Algorithm
This can be extended further for any multiplier with the following algorithm:

Compute the new period spacing 
�
1
=
�
�
�
new
P 
1
​
 = 
F 
new
​
 
F 
s
​
 
​
 

For every new epoch at 
�
=
{
�
1
,
2
�
1
,
3
�
1
,
…
 
}
i={P 
1
​
 ,2P 
1
​
 ,3P 
1
​
 ,…}:

Find the closest epoch in the original signal

Approximate the impulse response by applying a Hanning window of length 
2
�
0
+
1
2P 
0
​
 +1 centered at the original epoch

Overlap and add the windowed epoch into your new buffer centered at index 
�
i

Part 3 - Buffer Manipulation
TD-PSOLA can be done on an arbitrarily long signal (as you will do in Python), but to run this algorithm in real-time, we need to use buffers. Our autocorrelation-based pitch detector requires at least 40 ms buffers, and we do not want to add any more delay than we have to, so let's keep 40 ms buffers. Our setup then is:

bufferIn:  <-- 40 ms -->

bufferOut: <-- 40 ms -->
COPY
Consider what happens if we do not do anything to ensure buffer continuity. Overlap-added Hanning windows do permit a perfect reconstruction (if we try to generate a synthesis signal with P0 = P1), but note what happens when we have a non-integer number of epochs in our original signal:



Everything beyond sample index 16 is perfectly reconstructed, but the samples prior are missing information for perfect reconstruction. If we do this across multiple buffers, we get a reconstruction waveform with clipping at the boundaries:



This problem does not come up when pitch shifting an entire .wav file, because offline processing is allowed to be acausal. In other words, when doing this in Python, it is permissible to look into the future for your next epoch.

You cannot look into the future with online processing, but you can delay your buffer slightly to fake having access to past, present, and future data.

bufferIn:  | <-- 20 ms past --> | <-- 20 ms present --> | <-- 20 ms future --> |

bufferOut: | <-- 20 ms past --> | <-- 20 ms present --> | <-- 20 ms future --> |
COPY
Every time we get a new 20 ms frame of data, we shift it into the bufferIn queue as in Lab 2. Delaying our present buffer by 20 ms, we can then look "into the future" for our epoch computation.

The trick is then how we deal with bufferOut. For the TD-PSOLA algorithm itself, we only compute new epochs for the present buffer. However, we let the present epochs spill over into the past and future buffers when doing overlap-add.

For example, say we have the following bufferIn, and we have epochs at 0, 2, 4, 6, 8, 10, 12, 14 with P_0 = 1.

bufferIn:  0   1   2   3   4 | 5   6   7   8   9 | 10  11  12  13  14
COPY
If we want to perfectly reconstruct the signal, we can use the same epochs as in the original signal. Because we only compute present buffer epoch, we only compute overlap-add for epochs 6, 8. Assume that this iteration had run before, and the past buffer is already computed. (N) denotes an incomplete reconstruction.

bufferIn:  0   1   2   3   4 | 5   6   7   8   9 | 10  11  12  13  14

bufferOut: 0   1   2   3   4 |(5)  0   0   0   0 | 0   0   0   0   0
                          +win(5   6   7)
                                  +win(7   8   9)
         = 0   1   2   3   4 | 5   6   7   8  (9)| 0   0   0   0   0
COPY
Notice how 5, 6, 7, 8 are perfectly reconstructed, but 9 did not get enough information for complete reconstruction. Our past buffer is perfectly reconstructed, however, so we output that buffer to the speaker and shift our bufferOut accordingly. In the next iteration, we have the following buffer configuration:

bufferIn:  5   6   7   8   9 | 10  11  12  13  14| 15  16  17  18  19

bufferOut: 5   6   7   8  (9)| 0   0   0   0   0 | 0   0   0   0   0
COPY
Trying to reconstruct the present buffer again, we have the following:

bufferIn:  5   6   7   8   9 | 10  11  12  13  14| 15  16  17  18  19

bufferOut: 5   6   7   8  (9)| 0   0   0   0   0 | 0   0   0   0   0
                      +win(9   10  11)
                              +win(11  12  13)
                                      +win(13  14  15)
         = 5   6   7   8   9 | 10  11  12  13  14|(15) 0   0   0   0
COPY
By allowing our present computation to spill over into the past and future buffers, we can guarantee that by the end of the present computation, the past buffer will be fully reconstructed.

Buffer Manipulation Algorithm
To implement this practically, the algorithm is as follows. Let FRAME_SIZE be the number of samples for each 20 ms section, and let i point to your new epoch positions.

Initialize i = FRAME_SIZE so it points at the first index of the present section of bufferOut. Only do this in the first iteration.

While i < 2 * FRAME_SIZE, or while i is inside present:

Compute the overlap-add algorithm detailed above

Increment i such that i += P_1

Decrement i such that i -= FRAME_SIZE

Letting i spill over into the future buffer (which is what happens in the last iteration of the while loop), we maintain a pointer to where the first epoch of the next frame should go. Decrementing by FRAME_SIZE ensures continuity after shifting your past buffer out to the speaker.

Note
Your TA will go over this in more detail at the beginning of your lab section. Don't panic!
Python
Part 4 - TD-PSOLA in Python
Given an audio file, your task will be to shift the entire signal to a predetermined frequency. We have made the following simplifications to make things easier:

The entire signal will be voiced.

The epoch locations in the original signal are precomputed and given as epoch_marks_orig, a NumPy array containing the indices of each epoch marker.

TD-PSOLA can be computed over the entire signal, not just individual frames.

This leaves remapping, windowing, and overlap-adding for you to implement as described in the Overlap-Add Algorithm section. Your implementation will be expected to work for any synthesis frequency within the nominal human vocalization range 
[
100
�
�
,
600
�
�
]
[100Hz,600Hz].

Practically, your output should sound like a monotone version of your input, regardless of how high or low-pitched the input voice is.

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import scipy.io.wavfile as spwav
#from mpldatacursor import datacursor
import sys

plt.style.use('ggplot')

# Note: this epoch list only holds for "test_vector_all_voiced.wav"
epoch_marks_orig = np.load("test_vector_all_voiced_epochs.npy")
F_s, audio_data = spwav.read("test_vector_all_voiced.wav")
N = len(audio_data)

######################## YOUR CODE HERE ##############################
F_new = 420
new_epoch_spacing = ??

audio_out = np.zeros(N)

# Suggested loop
for i in range(0, N, new_epoch_spacing):
    # Your OLA code here
COPY
Note
A quick implementation note – earlier in the lab notes, we assumed some fixed P_0 for our entire signal and computed our window as length N = 2 * P_0 + 1. This assumes our signal is unchanging over time, which is clearly not true.

Instead, compute P_0 as the average of the distance between the nearest two epochs, ie P_0 = (epoch[i + 1] - epoch[i - 1]) / 2 for your current epoch i.

Assignment 1
Implement TD-PSOLA on the given test file using the starter Python code given. Try for various frequencies, F_new = 100, 200, 300, 400 etc.

This section will be worth 2 lab points.

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
Buffer Manipulation Supplemental Diagrams






CC BY-SA 4.0 Thomas Moon. Last modified: October 10, 2024. Website built with Franklin.jl and the Julia programming language.
