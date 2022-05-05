from matplotlib import pyplot as plt, patches
import math

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

def angle_plot(line1, line2, offset=1.0, color=None, origin=(0, 0),
len_x_axis=1, len_y_axis=1):
   xy1 = line1.get_xydata()
   xy2 = line2.get_xydata()
   slope1 = (xy1[1][1] - xy1[0][1]) / float(xy1[1][0] - xy1[0][0])
   angle1 = abs(math.degrees(math.atan(slope1)))
   slope2 = (xy2[1][1] - xy2[0][1]) / float(xy2[1][0] - xy2[0][0])
   angle2 = abs(math.degrees(math.atan(slope2)))
   theta1 = min(angle1, angle2)
   theta2 = max(angle1, angle2)
   angle = theta2 - theta1
   if color is None:
      color = line1.get_color()

   return patches.Arc(origin, len_x_axis * offset, len_y_axis * offset, 0, theta1, theta2, color=color, label=str(angle) + u"\u00b0")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

l1 = plt.Line2D([0, 1], [0, 4], linewidth=1, linestyle="-", color="green")
l2 = plt.Line2D([0, 4.5], [0, 3], linewidth=1, linestyle="-", color="red")

ax.add_line(l1)
ax.add_line(l2)

angle = angle_plot(l1, l2, 0.25)
ax.add_patch(angle)

plt.show()