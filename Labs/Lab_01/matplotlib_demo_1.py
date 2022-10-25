import matplotlib.pyplot as plt
 
# data for plotting
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y = [5, 7, 8, 1, 4, 9, 6, 3, 5, 2, 1, 8]
 
plt.plot(x, y)
 
plt.xlabel('x-axis label')
plt.ylabel('y-axis label')
plt.title('Matplotlib Example')
 
plt.savefig("out/output.jpg")