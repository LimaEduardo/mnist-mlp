import random
import math
import copy
import warnings
import numpy as np
import json
import time

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.datasets.base import get_data_home 
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



TRANING_DATASET_SIZE = 0.8
TEST_DATASET_SIZE = 0.2

warnings.filterwarnings("ignore", category=DeprecationWarning)
mnist_dataset = fetch_mldata('MNIST original')
data = mnist_dataset.data
target = mnist_dataset.target

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=TEST_DATASET_SIZE, random_state=42)
fig, ax = plt.subplots(2,5)
for i, ax in enumerate(ax.flatten()):
    im_idx = np.argwhere(target == i)[0]
    plottable_image = np.reshape(data[im_idx], (28, 28))
    ax.imshow(plottable_image, cmap='gray_r')

fig, ax = plt.subplots(1)
im_idx = 34567
plottable_image = np.reshape(data[im_idx], (28, 28))
ax.imshow(plottable_image, cmap='gray_r')


# for index, value in enumerate(data[im_idx]):
#     if index % 28 == 0: print("\n")
#     print(f"{value} ", end="")

# random.seed(123)

# Load file
def read_data_set(data_set):
    with open(data_set) as data_file:
        data_set = data_file.read().split("\n")
        for i in range(len(data_set)):
            data_set[i] = data_set[i].split(",")

    return data_set



# Convert string list to float list
def change_string_to_float(string_list):
    float_list = []
    for i in range(len(string_list)):
        float_list.append(float(string_list[i]))
    return float_list


# Matrix multiplication (for Testing)
def matrix_mul_bias(A, B, bias):
    C = []
    for i in range(len(A)):
        C.append([])
        for j in range(len(B[0])):
            C[i].append(0)
    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    
    return C


# Vector (A) x matrix (B) multiplication
def vec_mat_bias(A, B, bias):
    C = []
    for i in range(len(B[0])):
        C.append(0)
    
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
        C[j] += bias[j]
    
    return C


# Matrix (A) x vector (B) multipilicatoin (for backprop)
def mat_vec(A, B): 
    C = []
    for i in range(len(A)):
        C.append(0)
    
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    
    return C


# derivation of sigmoid (for backprop)
def sigmoid(A, deriv=False):
    if deriv: 
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

# Main funciton
if __name__=="__main__":



    output_data = {}

    train_set = data_train
    train_result = target_train


    test_set = data_test
    test_result = target_test






    # Define parameter
    alpha = 0.07
    epoch = 100
    neurons = [64, 128, 10] # number of neurons each layer







    # Initiate weight and bias with 0 value
    weights = []
    for i in range(len(neurons) - 1):
        weights.append([])
        for j in range(neurons[i]):
            weights[i].append([])
            for k in range(neurons[i + 1]):
                weights[i][j].append(0)
    
    weight = []
    for i in range(len(weights[0])):
        weight.append([])
        for j in range(len(weights[0][i])):
            weight[i].append(weights[0][i][j])

    weight_2 = []
    for i in range(len(weights[1])):
        weight_2.append([])
        for j in range(len(weights[1][i])):
            weight_2[i].append(weights[1][i][j])

    
    bias_list = []
    for i in range(1, len(neurons)):
        bias_list.append([])
        for j in range(neurons[i]):
            bias_list[i-1].append(0)

    bias = []
    for i in range(len(bias_list[0])):
        bias.append(bias_list[0][i])

    bias_2 = []
    for i in range(len(bias_list[1])):
        bias_2.append(bias_list[1][i])

    # Initiate weight with random between -1.0 ... 1.0
    for i in range(neurons[0]):
        for j in range(neurons[1]):
            weight[i][j] = 2 * random.random() - 1

    for i in range(neurons[1]):
        for j in range(neurons[2]):
            weight_2[i][j] = 2 * random.random() - 1


    e = 0
    has_time = True
    inicio = time.time()
    while(e < epoch and has_time):
        cost_total = 0
        for idx, data_list in enumerate(train_set): # Update for each data; SGD
            
            
            # Forward propagation
            h_1 = vec_mat_bias(data_list, weight, bias)
            X_1 = sigmoid(h_1)
            h_2 = vec_mat_bias(X_1, weight_2, bias_2)
            X_2 = sigmoid(h_2)
            


            # Convert to One-hot target
            target = [0] * neurons[-1]
            target[int(train_result[idx])] = 1


            # Cost function, Square Root Eror
            eror = 0
            for i in range(neurons[-1]):
                eror +=  0.5 * (target[i] - X_2[i]) ** 2 
            cost_total += eror

            # Backward propagation
            # Update weight_2 and bias_2 (layer 2)



            delta_2 = []
            for j in range(neurons[2]):
                delta_2.append(-1 * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))
            

            for i in range(neurons[1]):
                for j in range(neurons[2]):
                    weight_2[i][j] -= alpha * (delta_2[j] * X_1[i])
                    bias_2[j] -= alpha * delta_2[j]
            
            delta_1 = mat_vec(weight_2, delta_2)
            for j in range(neurons[1]):
                delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
            
            
            # Update weight and bias (layer 1)
            delta_1 = mat_vec(weight_2, delta_2)
            for j in range(neurons[1]):
                delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
            
            for i in range(neurons[0]):
                for j in range(neurons[1]):
                    weight[i][j] -=  alpha * (delta_1[j] * data_list[i])
                    bias[j] -= alpha * delta_1[j]
        
            
        cost_total /= len(train_set)
        if(e % 100 == 0):
            print(cost_total)
        
        fim = time.time()
        if ((fim - inicio) > (3600 * 1.25)):
            has_time = False

        e += 1

    
    print(fim, inicio, fim - inicio)
    res = matrix_mul_bias(test_set, weight, bias)
    res_2 = matrix_mul_bias(res, weight_2, bias_2)


    # Get prediction
    preds = []
    for r in res_2:
        preds.append(max(enumerate(r), key=lambda x:x[1])[0])


    # print(output_data["confusion_matrix"])


    for i in range(len(test_result)):
        test_result[i] = int(test_result[i])
    # Print prediction
    print("Resultado esperado: ", list(test_result))
    print("Predição:", preds)

    # Calculate accuration
    acc = 0.0
    for i in range(len(preds)):
        if preds[i] == int(test_result[i]):
            acc += 1
    acc = acc / len(preds)
    print(acc * 100, "%")




    output_data["total_training_time"] = fim - inicio
    fim = time.time()

    output_data["total_time"] = fim - inicio

    output_data["len_train"] = len(train_set)
    output_data["len_test"] = len(test_set)


    output_data["total_epochs"] = e
    output_data["weights_1"] = {}
    for pos, w in enumerate(weight):
        output_data["weights_1"][str(pos)] = w
    


    output_data["weights_2"] = {}
    for pos, w in enumerate(weight_2):
        output_data["weights_2"][str(pos)] = w
    
    output_data["bias_1"] = {}
    for pos, b in enumerate(bias):
        output_data["bias_1"][str(pos)] = b
    
    output_data["bias_2"] = {}
    for pos, b in enumerate(bias):
        output_data["bias_2"][str(pos)] = b
    
    output_data["predictions"] = preds
    output_data["target"] = list(test_result)
    cm = confusion_matrix(test_result,preds)
    output_data["confusion_matrix"] = {}

    for pos, item in enumerate(cm):
        output_data["confusion_matrix"][str(pos)] = []
        for i in range(len(item)):
            output_data["confusion_matrix"][str(pos)].append(int(item[i]))

    #     output_data["confusion_matrix"][str(pos)] = list(item)
    output_data["acc"] = acc

    # print(output_data)

    with open("resultados.json", "w") as output_file:
        output_file.write(json.dumps(output_data, indent=2))