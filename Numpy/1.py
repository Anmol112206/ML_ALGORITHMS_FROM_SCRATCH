import numpy as np
import math

a = np.array([10,2,3,5,4,2])
print(a[:3])

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(b[1,3])     #row with index 1 and column with index 3

print(a.ndim,b.ndim)   #prints the dimension of the array
print(b.shape)  #shape is a tuple specifying the number of elements along each dimension

print(b.size == math.prod(b.shape))    #size gives the total number of elements

print(a.dtype)   #prints the data type 


#Create a basic array
c = np.ones(2)
d = np.zeros(2)
e = np.empty(2)   #fills random values
f = np.arange(4)  #fills with a range of value up to 4
g = np.arange(2,9,2)  # first number ,last number (exclusive), step up
h = np.linspace(0,10,num=5)  #first number , last number(inclusive), total number of numbers in the array to be equally divided 
i = np.zeros(2,dtype = np.int64)  #default float64  , explicitly define int64
print(c,d,e,f,g,h,i)


#Adding, removing and sorting
j = np.array([3,2,7,3,6,1,5,6])
print(np.sort(j))

k = np.array([[1, 2], [3, 4]])
l = np.array([[5, 6]])
print(np.concatenate((c,d)))
print(np.concatenate((k,l), axis = 0))   #axis = 0 means column wise 


#Reshaping an array: Number of elements must remain same after reshaping
print(j.reshape(2,4))


#1D array to 2D array
m = np.array([1, 2, 3, 4, 5, 6])
m2 = m[np.newaxis,:]   #row vector 
m3 = m[:,np.newaxis]   #column vector
m4 = np.expand_dims(m,axis=1)
m5 = np.expand_dims(m,axis=0)
print(m2,m2.shape,m3,m3.shape,m4,m5)


#Indexing and Slicing
print(m[-2:])
n = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(n[n<5])   #prints all elements less than 5
o = np.array([1,2,3,4,5,6])
print(o[o<3],o<3)
five_up = (n >= 5)
print(n[five_up])
c = n[(n > 2) & (n < 11)]        #& is used for the set intersection
print(c)
print(np.nonzero(n < 5))  #returns the indices with value less than 5: first vector gives the row vector and second gives the column vector
print(n[np.nonzero(n<5)])
print(n[1:3],n[0:2,0])   #slice from index 0 to 2 (exclusive) in the 0th row


#Create a new array from existing one
p = np.array([[1, 1],[2, 2]])
q = np.array([[3, 3],[4, 4]])
print(np.vstack((p,q)))
print(np.hstack((p,q)))
r = np.arange(1, 25).reshape(2, 12)
print(np.hsplit(r, 3))
print(np.hsplit(r, (3, 4)))   #on spliting array from third and fourth column
s = r.copy()