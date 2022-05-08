import numpy as np

def print_errors_latex(error):
    m, _ = error.shape
    # min
    print("\\textbf{Min AE} [mm] &", end = "")
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(error[i][0], 2)) + "$" + "\\\\", end = "")
        else: 
            print("$" + str(round(error[i][0], 2)) + "$" + " &", end = "")
    print("\n")

    # max
    print("\\textbf{Max AE} [mm] &", end = "")
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(error[i][1], 2)) + "$" + "\\\\", end = "")
        else: 
            print("$" + str(round(error[i][1], 2)) + "$" + " &", end = "")
    print("\n")

    # RMSE 
    print("\\textbf{RMSE} [mm] &", end = "")
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(error[i][2], 2)) + "$" + "\\\\", end = "")
        else: 
            print("$" + str(round(error[i][2], 2)) + "$" + " &", end = "")
    print("\n")

    # estimation error 
    print("\\textbf{EE} [mm] &", end = "")
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(error[i][3], 2)) + " \\pm " + str(round(error[i][4], 2)) + "$" + "\\\\", end = "")
        else: 
            print("$" + str(round(error[i][3], 2)) + " \\pm " + str(round(error[i][4], 2)) + "$" + " &", end = "")

    print("\n")

    tmp = np.mean(error, axis = 0)
    for i in range(len(tmp)): 
        if i == m - 1:
            print("$" + str(round(tmp[i], 2)) + "$" + "\\\\", end = "")
        else: 
            print("$" + str(round(tmp[i], 2)) + "$" + " &", end = "") 

def print_angles(angles):
    m, _ = angles.shape
    diff = angles[:, 0] - angles[:, 1]
    # annotation
    print("Ann [$\\degree$] &", end = "")
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(angles[i][0], 2)) + "$ \\\\")
        else: 
            print("$" + str(round(angles[i][0], 2)) + "$ & ", end = "")

    # Model
    print("Model [$\\degree$] &", end = "")
    for i in range(m):
        if i == m - 1:
            print("$" + str(round(angles[i][1], 2)) + "$ \\\\")
        else: 
            print("$" + str(round(angles[i][1], 2)) + "$ & ", end = "")
    # Difference
    print("$|$Diff$|$ [$\\degree$] &", end = "")
    for i in range(m):
        if i == m - 1:
            print("$" + str(np.abs(round(diff[i], 2))) + "$ \\\\")
        else: 
            print("$" + str(np.abs(round(diff[i], 2))) + "$ & ", end = "")

    
