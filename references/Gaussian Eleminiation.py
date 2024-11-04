import numpy as np

def gaussElimination(a_matrix,b_matrix):
    # Adding some contingencies to prevent future problems 
    if a_matrix.shape[0] != a_matrix.shape[1]:
        print("ERROR: Square matrix not given!")
        return
    
    if b_matrix.shape[1] > 1 or b_matrix.shape[0] != a_matrix.shape[0]:
        print("ERROR: Constant vector incorrectly sized")
        return
     
    # Initialization of necessary variables
    n = len(b_matrix)
    m = n-1
    i = 0
    j = i-1
    x = np.zeros(n)
    new_line = "\n"
    
    # Create our augmented matrix through NumPys concatenate feature and force data type to float.
    augmented_matrix = np.concatenate((a_matrix, b_matrix,), axis=1, dtype=float)
    print(f"The initial augmented matrix is: {new_line}{augmented_matrix}")
    print("Solving for the upper-triangular matrix:")
    
    
    # Applying Gaussian Elimination w/ Pivoting:
    while i < n:
        
        # Partial Pivoting
        for p in range(i+1, n):
            if abs(augmented_matrix[i,i]) < abs(augmented_matrix[p, i]):
                augmented_matrix[[p, i]] = augmented_matrix[[i, p]]
                
        if augmented_matrix[i, i] == 0.0: #fail-safe to eliminate divide by zero error!
            print("Divide by zero error")
            return
        
        for j in range(i+1,n):
            scaling_factor = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j] = augmented_matrix[j] - (scaling_factor * augmented_matrix[i])
            print(augmented_matrix) # To visualize each elimination step.
            
        i= i + 1
    
        # Backwards substitution to solve for x-matrix:
        x[m] = augmented_matrix[m][n] / augmented_matrix[m][m]
        
        for k in range(n-2,-1,-1):
            x[k] = augmented_matrix[k][n]
            
            for j in range(k+1, n):
                x[k] = x[k] - augmented_matrix[k][j] * x[j]
                
            x[k] = x[k] / augmented_matrix[k][k]
    
    # Displaying solution 
    print("The following x-vector matrix solves the above augmented matrix:")
    for answer in range(n):
        print(f"x{answer} is {x[answer]}")


if __name__ == "__main__":
    variable_matrix = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 9, 3, 1, 0, 0, 0],
                                [9, 3, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 25, 5, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 25, 5, 1],
                                [0, 0, 0, 0, 0, 0, 64, 8, 1],
                                [6, 1, 0, -6, -1, 0, 0, 0, 0],
                                [0, 0, 0, 10, 1, 0, -10, -1, 0],
                                [2, 0, 0, 0, 0, 0, 0, 0, 0]])
    constant_matrix = np.array([[2],
                                [3],
                                [3],
                                [9],
                                [9],
                                [10],
                                [0],
                                [0],
                                [0]])

    gaussElimination(variable_matrix, constant_matrix)
    
    solution = np.linalg.solve(variable_matrix, constant_matrix)
    print(np.round(solution, 4))