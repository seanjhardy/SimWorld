import numpy as np
import math

def encode_scalar(input, minVal, maxVal, length, overlapPercent=1):
    input = min(max(minVal, input), maxVal)
    input = (input - minVal) / (maxVal - minVal)
    output = np.zeros(length, dtype=int)
    if overlapPercent == 1:
        index = max(min(int(length * input), length-1), 0)
        output[index] = 1
    else:
        firstIndex = max(min(int(length * (input - overlapPercent*0.5)), length), 0)
        lastIndex = max(min(int(length * (input + overlapPercent*0.5)), length), 0)
        if (firstIndex == lastIndex):
            output[firstIndex] = 1
        else:
            output[firstIndex:lastIndex] = 1
    return output
    
def decode_scalar(inputArr, minVal, maxVal, length, overlapPercent=1):
    if np.sum(inputArr) == 0:
        return 0

    if overlapPercent == 1:
        input_approx = np.argmax(inputArr)/ (length - 1)
        input_approx = input_approx * (maxVal - minVal) + minVal
        return input_approx
    
    active_indices = np.where(inputArr == 1)[0]
    first_index = active_indices[0]
    last_index = active_indices[-1] + 1
    
    index_range = last_index - first_index
    missing_bits = int(overlapPercent * length - index_range)
    if first_index == 0:
        active_indices = np.concatenate((np.arange(-missing_bits+1, 1), active_indices))
    if last_index == length:
        active_indices = np.concatenate((active_indices, np.arange(length, length+missing_bits)))

    input_approx = np.mean(active_indices) / (length - 1)
    input_approx = input_approx * (maxVal - minVal) + minVal
    return min(max(minVal, input_approx), maxVal)

def encode_log(input, length, overlapPercent, negative=True):
    #range = -inf to + inf
    input = 2/math.pi * math.atan(input) #-1 to + 1
    if negative:
        input = (input + 1) / 2 # 0 to 1
    output = np.zeros(length, dtype=int)
    firstIndex = max(min(int(length * (input - overlapPercent*0.5)), length), 0)
    lastIndex = max(min(int(length * (input + overlapPercent*0.5)), length), 0)
    if (firstIndex == lastIndex):
        output[firstIndex] = 1
    else:
        output[firstIndex:lastIndex] = 1
    return output

def decode_log(input_array, length, overlapPercent):
    active_indices = np.where(input_array == 1)[0] + 1

    if len(active_indices) == 0:
        return 0

    first_index = active_indices[0]
    last_index = active_indices[-1]
    index_range = last_index - first_index
    missing_bits = int(overlapPercent*length - index_range)

    if (missing_bits > 0):
        if first_index == 0:
            active_indices = np.append(active_indices, [x - (missing_bits-1) for x in range(0,missing_bits)])
        if last_index == length:
            active_indices = np.append(active_indices, [x for x in range(length,length + missing_bits)])
    active_indices = np.array(active_indices, dtype=np.float32)
    active_indices -= length * 0.5 + 1
    input_approx = (sum(active_indices)) / (len(active_indices) * (length-1))
    output = math.tan(input_approx * math.pi / 2)
    return output