import numpy as np
import matplotlib.pyplot as plt
import os

path="./pca_data"
i=0
a=0
x=[0,1,2,3,4,5,6,7,8,9]
data=[[]]
for file in os.listdir(f"{path}"):
    if file.endswith('explained_variance.npy'):
        a+=1
        if i<10: 
            i+=1
            data[len(data)-1].append((np.load(f"{path}/{file}")).tolist())
        else:
            i=1
            data.append([])
            data[len(data)-1].append((np.load(f"{path}/{file}")).tolist())

# print(data)
avarage_data=[[],[],[],[]]
for p in range(0,6):
    print_data=[]
    a=0
    for i in range(0,4):
        print_data.append([])
        for j in range(0,10):
            print_data[i].append(data[p][j][i])
            a+=data[p][j][i]
        avarage_data[i].append(a/10.0)
        a=0
        

    plt.plot(x, print_data[0], marker='s', linestyle='-', color='b', label='P1')
    plt.plot(x, print_data[1], marker='s', linestyle='-', color='y', label='P2')
    plt.plot(x, print_data[2], marker='s', linestyle='-', color='g', label='P2')
    plt.plot(x, print_data[3], marker='s', linestyle='-', color='r', label='P2')


    plt.title('Principal components variance %')
    plt.xlabel(f'Entry number for person number {p}')
    plt.ylabel('%')
    plt.legend()

    plt.savefig(f'./pictures_pca_variance/pca_variance_person_{p}.png')
    plt.clf()

print(avarage_data)
x=[1,2,3,4,5,6]
plt.plot(x, avarage_data[0], marker='s', linestyle='-', color='b', label='P1')
plt.plot(x, avarage_data[1], marker='s', linestyle='-', color='y', label='P2')
plt.plot(x, avarage_data[2], marker='s', linestyle='-', color='g', label='P2')
plt.plot(x, avarage_data[3], marker='s', linestyle='-', color='r', label='P2')


plt.title('Principal components variance %')
plt.xlabel('Person nr.')
plt.ylabel('%')
plt.legend()

plt.savefig(f'./pictures_pca_variance/pca_variance_avarage.png')
plt.clf()