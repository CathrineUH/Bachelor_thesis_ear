def print_errors_latex(error):
    m, _ = error.shape
    # min
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(error[i][0], 2)) + "$" + "\\\\", end = "")
        else: 
            print("$" + str(round(error[i][0], 2)) + "$" + " &", end = "")
    print("\n")
    # max
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(error[i][1], 2)) + "$" + "\\\\", end = "")
        else: 
            print("$" + str(round(error[i][1], 2)) + "$" + " &", end = "")

    print("\n")
    # RMSE 
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(error[i][2], 2)) + "$" + "\\\\", end = "")
        else: 
            print("$" + str(round(error[i][2], 2)) + "$" + " &", end = "")
        
    print("\n")
    # estimation error 
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(error[i][3], 2)) + " \\pm " + str(round(error[i][4], 2)) + "$" + "\\\\", end = "")
        else: 
            print("$" + str(round(error[i][3], 2)) + " \\pm " + str(round(error[i][4], 2)) + "$" + " &", end = "")


def print_angles(angles):
    m, _ = angles.shape
    diff = angles[:, 0] - angles[:, 1]
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(angles[i][0], 2)) + "$ \\\\")
        else: 
            print("$" + str(round(angles[i][0], 2)) + "$ & ", end = "")

    for i in range(m):
        if i == m - 1:
            print("$" + str(round(angles[i][1], 2)) + "$ \\\\")
        else: 
            print("$" + str(round(angles[i][1], 2)) + "$ & ", end = "")

    for i in range(m):
        if i == m - 1:
            print("$" + str(round(diff[i], 2)) + "$ \\\\")
        else: 
            print("$" + str(round(diff[i], 2)) + "$ & ", end = "")

    
