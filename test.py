from color_dtu_design.color_design import *
import matplotlib.pyplot as plt

# x axis values
x = [1,2,3,4,5,6]
# corresponding y axis values
y = [2,4,1,5,2,6]


z = color_design("pink").color
# plotting the points
plt.plot(x, y, color = z, linestyle='dashed', linewidth = 3,
		marker='o', markerfacecolor=z, markersize=12)
 
# setting x and y axis range
plt.ylim(0,8)
plt.xlim(0,8)

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('Some cool customizations!')

# function to show the plot
plt.show()
