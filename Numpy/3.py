import numpy as np
import matplotlib.pyplot as plt

#Printing docstring of arrays and functions 
#In jupyter, a?  and   a?? is enough to print while normally help is used
a = np.array([1,2,3,4,5])
#print(help(a))


#Working with mathematical formulas
n = 3
predictions = np.ones(3)
labels = np.arange(1,4,1)
error = (1/n)*np.sum(np.square(predictions-labels))
print(error)


#Save and load the objects
#load and save stores in the npy while savez stores in the npz extension
#for one: use npy while for n use npz 
np.save('text', a)   
b = np.load('text.npy')  #loads the array from the file
print(b)
np.savetxt('file.csv', a, fmt='%.2f', delimiter=',', header='1,  2,  3,  4')
print(np.loadtxt('file.csv'))


#Plot the indev vs value graph using matplotlib
c = np.array([2, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22])
#plt.plot(c)
plt.title("Simple Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
x = np.linspace(0, 5, 20)
y = np.linspace(0, 10, 20)
plt.plot(x, y, 'purple') # line is given the purple colour
plt.plot(x, y, 'o')      # dots is marked as o
plt.show()