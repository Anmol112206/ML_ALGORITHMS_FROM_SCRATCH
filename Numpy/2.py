import numpy as np

#Basic array operations
a = np.array([1, 2])
b = np.ones(2,dtype=int)
print(a + b)
print(a.sum(),a.min(),a.max())
c = np.array([[1, 1], [2, 2]])
print(c.sum(axis = 1))   #axis = 1 means column wise (gives sum of each subarrays)
print(c.sum(axis = 0))   #axis = 0 means row wise 


#Creating random numbers array
rng = np.random.default_rng() 
print(rng.random(3),rng.random((3,2)))
print(rng.integers(5, size=(2, 4)))    #generate a 2*4 array with integers in the range of 0 to 4


#Get unique items and counts
d = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
unique_values,indices_list,occurrence_count= np.unique(d,return_index = True,return_counts = True)
print(unique_values,indices_list,occurrence_count)   #indices_list prints the index of the first occurrence of all the unique elements
#in 2D array unique refers to unique rows 
e = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
unique_rows, indices, occurrence_count = np.unique(e, axis=0, return_counts=True, return_index=True)  
print(unique_rows, indices, occurrence_count)


#Transpose of the matrix
f = np.array([[0, 1, 2],[3, 4, 5]])
print(f.transpose(),f.T)


#Reverse a matrix: reverses the rows and columns simultaneously of 2D array
print(np.flip(f))
print(np.flip(f,axis=0))   #only reversing the rows
f[1] = np.flip(f[1])
print(f)   #reverses the contents of the first row of matrix f


#Reshaping and flattening multi-dimensional arrays
#flatten and ravel are two methods: ravel is actually a reference to the parent array and any changes in it leads to changes in the parent array, it is memory efficient
#while using flatten , no changes occur in the parent array
g = e.flatten()
h = e.ravel()
g[0] = 22
print(e,g)   
h[0] = 22
print(e,h)   