import numpy as np
import time
#np.array([1,2,3])

arr = np.array([1,2,3])
print(arr,type(arr))

#comparision of list and numpy array
size=100000

list1 = list(range(size))
arr1 = np.array(list1)

start = time.time()
for i in range(size):
    list1[i] = list1[i]*2
    
end = time.time()
print("Time taken by list: ", end-start)

start = time.time()
arr1 = arr1*2
end = time.time()
print("Time taken by numpy array: ", end-start)


#prefilled arrays
zeros = np.zeros((2,3))
print(zeros,zeros.shape,type(zeros))

#type casting
arr2 = np.array([1,2,3], dtype=np.float64)
print(arr2,arr2.dtype)

#reshaping arrays
arr3 = np.array([[1,2,3],[4,5,6]])
reshape_arr3 = arr3.reshape(3,2)
print(arr3,"\n",reshape_arr3)

#flattened array
flatten_arr3 = arr3.flatten()
print(flatten_arr3)

#indexing and slicing
print(flatten_arr3[0]) 
print(flatten_arr3[1:4])

#fancy indexing
arr4 = np.array([10,20,30,40,50])
print(arr4[[0,2,4]])


arr5=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr5[[0,2],[0,2]]) #fancy indexing with 2D array


arr6=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr6[0:2,1:3]) #slicing with 2D array

#Boolean indexing
arr7 = np.array([1,2,3,4,5])
print(arr7[arr7>3]) #Boolean indexing to filter elements greater than 3
print(arr7[arr7%2==0]) #Boolean indexing to filter even numbers


#slicing with step
arr8 = np.array([1,2,3,4,5,6,7,8,9])
print(arr8[::2]) #slicing with step to get every second element
print(arr8[1::2]) #slicing with step to get every second element starting from index 1
print(arr8[::-2]) #slicing with step to reverse the array
print(arr8[1:4])
print(arr8[1:4:2]) #slicing with step to get every second element from index 1 to 3
print(arr8[1:]) #slicing to get all elements from index 1 to the end
print(arr8[:5]) #slicing to get all elements from the start to index 4
print(arr8[::4]) #slicing with step to get every fourth element
print(arr8[1:8:3]) #slicing with step to get every third element from index 1 to 7

#broadcasting
arr9 = np.array([1,2,3])
print(arr9 + 10) #broadcasting to add 10 to each element of the array
arr10 = np.array([[1,2,3],[4,5,6]])
print(arr10 + np.array([10,20,30])) #broadcasting to add a 1D array to a 2D array

#copy vs view
arr11 = np.array([1,2,3])
arr12 = arr11 #this creates a view of arr11
arr13 = arr11.copy() #this creates a copy of arr11
arr12[0] = 10
arr13[1] = 20
print(arr11) #arr11 is modified because arr12 is a view of arr11
print(arr11) #arr13 is not modified because arr13 is a copy of arr11



