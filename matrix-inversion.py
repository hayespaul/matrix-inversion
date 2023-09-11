import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt
import timeit
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def matrixminor(m,i,j): # i column j row
    """
    returns an (n-1,n-1) size array with ith column and jth row deleted
    """
    x=np.delete(m, i, 1) # deletes ith row of matrix m
    y=np.delete(x, j, 0) # deletes jth column of matrix x
    return y

def determinant(m):
    """
    computes the determinant of arbitrary nxn input matrix m 
    """    
    if len(m) == 2: 
        return m[0,0]*m[1,1]-m[0,1]*m[1,0]  # returns ad-bc for a 2x2 matrix
    det = 0
    for k in range(len(m)):
        det += ((-1)**k)*m[0,k]*determinant(matrixminor(m,k,0))
    return det  

def matrixinverse(m):
    """
    Inverts an arbitrary nxn square matrix
    
    """          
    if len(m) == 1:
        return 1/np.sum(m)  # returns 1/number if length 1x1
    elif len(m) == 2:
        det = determinant(m) # just computes determinant for a 2x2 
        return m/det
    det = determinant(m)      
    cofactors = np.ndarray(shape=(len(m),len(m)),  dtype=float, order='F')    # cofactor array
    for k in range(len(m)):
        for l in range(len(m)):
            cofactors[k,l]=((-1)**(k+l))*determinant(matrixminor(m,k,l)) # assigns cofactor[k,l] using the determinant at a given point
    return cofactors/det



    
def solve_sim_LU(A,C):
    """
    Gives x vector by solving the equation Ax=C using LU decomposition
    """
    LU=sc.lu_factor(A)    
    xLU=sc.lu_solve(LU,C)    
    return xLU
    

def solve_sim_analytical(A,C):
    """
    Gives x vector by solving the equation Ax=C using the analytical method to invert the matrix
    """
    xA=np.dot(matrixinverse(A),C) # dot product        
    return xA

def solve_sim_svd(A,C):
    """
    Gives x vector by solving the equation Ax=C using SVD
    """
    U,D,V=sc.svd(A)
    z=np.dot(np.transpose(U),C) # Z is the dot product of the transpose of U with the vector C
    k=sc.solve(np.diag(D),z) # gives a matrix by multiplying the diagonal of D by Z
    xSVD=np.dot(np.transpose(V),k)  # gives x vector from the dot product of transpose of V with K
    return xSVD





def resolve_matrix(x,y,z):        
    """
    Outputs a 3x3 square matrix for the resolved values of matrrix A using inputed (x,y,z) of the camera's position
    """
    array = np.ndarray(shape=(3,3), dtype=float, order='F')    
    
    x1,y1=0,0    # position of Pillar 1
    x2,y2=90*np.sqrt(3),0       # pillar 2
    x3,y3=90*np.sqrt(3)/2,135 # Pillar 3    
                    
    x1rel=x-x1
    x2rel=x-x2
    x3rel=x-x3
              
    y1rel=y-y1
    y2rel=y-y2
    y3rel=y-y3
    
    d1=(np.sqrt((x1rel)**2 + (y1rel)**2))
    d2=(np.sqrt((x2rel)**2 + (y2rel)**2))
    d3=(np.sqrt((x3rel)**2 + (y3rel)**2))
    
    l1=np.sqrt(d1**2 + 49)
    l2=np.sqrt(d2**2 + 49)
    l3=np.sqrt(d3**2 + 49)      
    
    if x1rel == 0 or x2rel == 0 or y1rel == 0 or y2rel == 0 or y3rel==0 or x3rel==0: # if any= 0 do not compute sums as will get infs or NaNs
        return np.zeros((3,3))
    else:        
        array[0,0], array[0,1], array[0,2] = ((x1rel/d1), (x2rel/d2), (x3rel/d3))
        array[1,0], array[1,1], array[1,2] = ((y1rel/d1), (y2rel/d2), (y3rel/d3))   
        array[2,0], array[2,1], array[2,2] = ((z/l1), (z/l2), (z/l3))        
        return array


def solve_sim_task3(A):
    """
    Solves the sims for task 3
    """
    if np.any(A) == 0: # returns array of 0s if any = 0 so the solvers do not handle 0s. 
        return ((0,0,0)),((0,0,0)),((0,0,0))  
    
    C=np.array([0,0,-50*9.81])      
    
    LU=sc.lu_factor(A)    
    xLU=sc.lu_solve(LU,C)
    
    U,D,V=sc.svd(A)
    z=np.dot(np.transpose(U),C) 
    k=sc.solve(np.diag(D),z)
    xSVD=np.dot(np.transpose(V),k)

    xA=np.dot(matrixinverse(A),C)    
    
    return xA,xLU,xSVD


def plottask3():       
    """
    Plots intensity and 3D plots of the tension as a function of position of camera (x,y)
    """
    pitch_width=90*np.sqrt(3) # width of football pitch
    pitch_height=135     #height
    
    resolution_x=100 #granularity of plot
    resolution_y=100
    
    width_mesh=np.linspace(0,pitch_width,resolution_x)
    height_mesh=np.linspace(0,pitch_height,resolution_y) 
    
    xx,yy = np.meshgrid(width_mesh,height_mesh)     # define meshgrid of values for intensity plot
    
    tension_grid_1=np.zeros((resolution_x,resolution_y))  # tensions in wires from pillar 1
    tension_grid_av=np.zeros((resolution_x,resolution_y)) # average of the 3 wires tensions
    tension_grid_2=np.zeros((resolution_x,resolution_y))
    tension_grid_3=np.zeros((resolution_x,resolution_y))
    
    for i in range (resolution_x):
        for j in range (resolution_y):
            
            if np.arctan(yy[i,j]/xx[i,j]) < np.pi/3 and np.arctan(yy[i,j]/(90*np.sqrt(3) - xx[i,j])) < np.pi/3: # bounds the plot to inside allowable values of tension
                
                resolvedmatrix = resolve_matrix(xx[i,j], yy[i,j],-7) #gives matrix for camera x,y
                tensionsA,tensionsLU,tensions_svd=solve_sim_task3(resolvedmatrix)  # gives the 3 tensions T1,T2,T3
                tension_grid_1[i,j]= abs(tensionsLU[0])  # assigns T1 to tension_grid i,j   
                tension_grid_2[i,j]= abs(tensionsLU[1])     
                tension_grid_3[i,j]= abs(tensionsLU[2]) 
                            
                            
                tension_grid_av[i,j]= abs(np.sum(tensionsLU)/3)  # average of T1 T2 T3
                
                                                         
    ax = Axes3D(plt.figure())
    ax.plot_surface(xx,yy,tension_grid_av, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0)
    plt.title('Surface plot of average tension in 3 wires')
    plt.show()
    
    ax.plot_surface(xx,yy,tension_grid_1, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0)
    plt.title('Surface plot in Tension 1.')
    plt.show()
    
    plt.pcolormesh(xx,yy,tension_grid_1, cmap='hot')
    plt.colorbar()
    plt.title('Intensity plot of Tension 1')
    plt.show()  
    
    plt.pcolormesh(xx,yy,tension_grid_2, cmap='hot')
    plt.colorbar()
    plt.title('Intensity plot of Tension 2')
    plt.show()  
    
    plt.pcolormesh(xx,yy,tension_grid_3, cmap='hot')
    plt.colorbar()
    plt.title('Intensity plot of Tension 3')
    plt.show()  
    
    plt.pcolormesh(xx,yy,tension_grid_av, cmap='hot')
    plt.colorbar()
    plt.show()   
    plt.title('Intensity plot of average of three tensions. ')
    plt.pause(1)

    
    Tmax=np.max(tension_grid_1)
    print("")
    print(" The maximum tension in T1 is",(Tmax),"N.")  
    print("This occurs at:")
    
    N=100
    
    x_value=np.linspace(0,156,N)
    y_value=np.linspace(0,134,N)
    
    T=tension_grid_1
    
    for i in range(len(T)):
        for j in range(len(T)):
            if (round(T[i,j],8) == round(Tmax,8)): # when T = Tmax
                print('x=', round(x_value[i],2),'m') # prints current x value
                print('y=', round(y_value[j],2),'m')
    
    return





def timing_function(function,matrix,n):
    """
    Times how long a given function takes to produce output. Output of function as n size array, not averaged. 
    """
    total_n=np.zeros(n)
    for i in range(n):        
        start_time = timeit.default_timer()
        function(matrix)
        total_n[i]=timeit.default_timer() - start_time
    return total_n # array of values for each run



def plot_rowsvstime_task1(maxsize,n):  
    """
    Plots computational time against number of rows for computing the inverse in task 1.
    """
    matrixsize=np.arange(1,maxsize+1,1) # size of matrix
    time=np.zeros(len(matrixsize))
    error=np.zeros(len(matrixsize)) # error bars
    
    for i in range(len(matrixsize)):       
        matrix=np.random.randint(0,9,(matrixsize[i],matrixsize[i]))  # random matrix
        times= timing_function(matrixinverse,matrix,n) # gives array of times
        time[i]= np.sum(times)/n  # averages the array of times
        error[i] = np.max(times)-np.min(times)      # standard deviation

    plt.errorbar(matrixsize, time, error, color='r', linestyle='-', marker='.',capsize=4)
    plt.xlabel("Rows")
    plt.ylabel("Time(s)")    
    plt.show()
    plt.pause(1)
    
    return



def timing_function_2(function,matrix,c,n):
    """
    Used in timing LU and SVD, output is array of times.
    """
    total_n=np.zeros(n)
    for i in range(n):        
        start_time = timeit.default_timer()
        function(matrix,c)
        total_n[i]=timeit.default_timer() - start_time
    return total_n

def test_speeds_2(maxsize,n):
    """
    Tests the speed of LU and SVD at solving sim equations, plots a graph of values.
    """
    matrixsize=np.arange(2,maxsize+1,1)
    time_svd=np.zeros(len(matrixsize))   
    time_LU=np.zeros(len(matrixsize)) # times for LU to solve sim equation
    error_svd=np.zeros(len(matrixsize)) # standard deviation in svd
    error_LU=np.zeros(len(matrixsize))    
    
    for i in range(len(matrixsize)):       
        matrix=np.random.randint(0,9,(matrixsize[i],matrixsize[i])) # random nxn matrix
        C=np.random.randint(0,9,(matrixsize[i])) #random matrix C
        
        times_svd = timing_function_2(solve_sim_svd,matrix,C,n) # array of times for SVD
        times_LU = timing_function_2(solve_sim_LU,matrix,C,n)
        
        time_svd[i]=np.sum(times_svd)/n # average times for SVD for a given matrix size
        time_LU[i]=np.sum(times_LU)/n
               
        error_svd[i]=np.max(times_svd) - np.min(times_svd) # standard deviations for SVD for ith matrix size
        error_LU[i]=np.max(times_LU) - np.min(times_LU)
    
    plt.errorbar(matrixsize, time_svd, error_svd, color='r', linestyle='-',capsize=4,errorevery=20,label=("SVD"))
    plt.errorbar(matrixsize, time_LU, error_LU, color='b', linestyle='-',capsize=4,errorevery=20, label=("LU"))
    plt.legend(loc='best')
    plt.xlabel("Rows")
    plt.ylabel("Time(s)") 
    plt.show()
    plt.pause(1)
    return
    
def test_singular():
    """
    """
    C1=5
    C2=10
    C3=15
    
    n=3
    array = np.ndarray(shape=(n,n), dtype=float, order='F')
    
    k=np.linspace(0.001,1,100)
    
    for i in range (len(k)):       
    
        array[0,0], array[0,1], array[0,2] = (1,1,1)
        array[1,0], array[1,1], array[1,2] = (1,2,-1)   
        array[2,0], array[2,1], array[2,2] = (2,3,k[i]) 
        
        print(k[i])
        
        xLU=solve_sim_LU(array,C1,C2,C3)
        xsvd=solve_sim_svd(array,C1,C2,C3)
        xA=solve_sim_analytical(array,C1,C2,C3)
        print(xLU)  
        print(xsvd)
        print(xA)
    return


def test_matrixinverse():
    """
    Menu system for Task 1.
    """
    MyInput = '0'
    while MyInput != 'q': 
        print('-------------------------------MENU-------------------------------------------')
        print("[1]Test 1: Performs the inverse of a 1x1 matrix.")
        print("[2]Test 2: Performs the inverse of a known 2x2 matrix.")
        print("[3]Test 3: Performs the inverse of a known 3x3 matrix.")
        print("[4]Test 4: Performs the inverse of a known 4x4 matrix.")
        print("[5]Test 5: Performs the inverse of a known 5x5 matrix.")
        print("[6]Test 6: Performs the inverse of a random nxn matrix of the user's choice")
        print("[7]Test 7: Plots a graph of size of matrix against time.")
        print("-------------------------------------------------------------------------------")
        MyInput = input('Type 1, 2, 3, 4, 5, 6, 7 or q to exit:')
        if MyInput == '1':        
            print("\n######## TEST 1 #########")            
            MyInput2= float(input('What number would you like to inverse?'))       
            print("\nThe inverse of", MyInput2, "is", matrixinverse([MyInput2]))
            print("\n##########################")           
        elif MyInput == '2':
            print("\n######## TEST 2 #########")             
            n=2
            array = np.ndarray(shape=(n,n), dtype=float, order='F')            
            array[0,0], array[0,1] = (1,3)
            array[1,0], array[1,1] = (4,5)
            
            print("\nThe inverse of: ")
            print("")
            print(array)
            print("")
            print("is:")
            print("")
            print(matrixinverse(array))
            print("")
            print("\n##########################")
        elif MyInput == '3':
            print("\n######## TEST 3 #########")
            n=3
            array = np.ndarray(shape=(n,n), dtype=float, order='F')
            array[0,0], array[0,1], array[0,2] = (3,4,5)
            array[1,0], array[1,1], array[1,2] = (1,2,5)   
            array[2,0], array[2,1], array[2,2] = (4,6,3)        
            
            print("\nThe inverse of: ")
            print("")
            print(array)
            print("")
            print("is:")
            print("")
            print(matrixinverse(array))
            print("")
            print("\n##########################")
        elif MyInput == '4':
            print("\n######## TEST 4  #########")            
            n=4        
            array = np.ndarray(shape=(n,n), dtype=float, order='F')
            
            array[0,0], array[0,1], array[0,2], array[0,3] = (5,6,4,2)
            array[1,0], array[1,1], array[1,2], array[1,3] = (3,2,1,4)     
            array[2,0], array[2,1], array[2,2], array[2,3] = (6,1,3,2)
            array[3,0], array[3,1], array[3,2], array[3,3] = (6,2,8,1)
            
            print("\nThe inverse of: ")
            print("")
            print(array)
            print("")
            print("is:")
            print("")
            print(matrixinverse(array))
            print("")
            print("\n##########################")       
        elif MyInput == '5':
            print("\n######## TEST 5  #########")            
            n=5        
            array = np.ndarray(shape=(n,n), dtype=float, order='F')
            
            array[0,0], array[0,1], array[0,2], array[0,3], array[0,4] = (1,0,0,0,0)
            array[1,0], array[1,1], array[1,2], array[1,3], array[1,4] = (0,1,0,0,-1)     
            array[2,0], array[2,1], array[2,2], array[2,3], array[2,4] = (0,3,0,-3,-2)
            array[3,0], array[3,1], array[3,2], array[3,3], array[3,4] = (0,0,2,-1,0)
            array[4,0], array[4,1], array[4,2], array[4,3], array[4,4] = (1,-1,0,1,0)  
            
            print("\nThe inverse of: ")
            print("")
            print(array)
            print("")
            print("is:")
            print("")
            print(matrixinverse(array))
            print("")
            print("\n##########################")
        elif MyInput == '6':
            print("\n######## TEST 6  #########")
            MyInput = int(input('What size matrix would you like to invert? (A value for the dimension NxN of the matrix)'))           
            array=np.random.randint(0,9,(MyInput,MyInput))
            
            print("\nThe inverse of: ")
            print("")
            print(array)
            print("")
            print("is:")
            print("")
            print(matrixinverse(array))
            print("")
            print("\n##########################")        
        elif MyInput == '7':
            print("\n######## TEST 7  #########")
            print("")
            print("Plotting graph of 9 rows with 3 repeats, please wait.")
            print("Could take several minutes....")      
            
            plot_rowsvstime_task1(9,3)
            
            print("")
            print("\n##########################") 
            
            
        else:
            print("")        
    print('Returning to main menu...')    
    
    return

def test_task2():
    """
    Menu system for task 2.
    """
    MyInput = '0'
    while MyInput != 'q': 
        print('-------------------------------MENU--------------------------------------------------------------------')
        print("[1]Test 1: Plots a graph comparing the speeds of svd and LU for a random set of N simultaneous equations.")
        print("-------------------------------------------------------------------------------")
        MyInput = input('Type 1 or q to return to MAIN MENU:')
        if MyInput == '1':        
            print("\n######## TEST 1 #########")       
            
            test_speeds_2(500,5)
            

            print("\n##########################")           
        elif MyInput == '2':
            print("\n######## TEST 2 #########")             
            
            print("")
            print("\n##########################")            
            
        else:
            print("")        
    print("")
    print('Returning to main menu....')    
    
    
    
    return


def main_menu():
    """
    Menu system for overall code.
    """
    MyInput = '0'
    while MyInput != 'q': 
        print("")
        print('------------------------------- MAIN MENU ------------------------------------------------------------')
        print("[1]Task 1: Tests the algorithm for inverting an NxN matrix for a few known matrices.")
        print("[2]Task 2: Invesigates LU, SVD and the analytical method for solving simultenous equations.")
        print("[3]Task 3: Plots graphs for the tension in 3 wires suspending a camera.")
        print("------------------------------------------------------------------------------------------------------")
        print("")
        MyInput = input('Select 1, 2, 3 or q to quit:')
        if MyInput == '1':        
            print("\n######## TASK 1 #########")
            test_matrixinverse()

        elif MyInput == '2':
            print("\n######## TASK 2 #########")
            test_task2()


        elif MyInput == '3':
            print("\n######## TASK 3 #########")   
            plottask3()
        else:
            print("")        
    print('Goodbye')    
    
    return
    
main_menu()

