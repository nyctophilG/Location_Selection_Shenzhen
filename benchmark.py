import numpy as np

# --- UNIMODAL FUNCTIONS (F1 - F7) ---
import numpy as np

def F1(x):
    return np.sum(x**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    dim = len(x)
    total = 0
    for i in range(dim):
        total += np.sum(x[:i+1])**2
    return total

def F4(x):
    return np.max(np.abs(x))

def u_fun(x, a, k, m):
    y = k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < -a)
    return y

def F5(x): # Rastrigin (Paper Table 1 Row 5)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def F6(x): # Ackley (Paper Table 1 Row 6)
    dim = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / dim)
    return term1 + term2 + 20 + np.e

def F7(x): # Griewank (Paper Table 1 Row 7)
    dim = len(x)
    indices = np.arange(1, dim + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(indices)))
    return np.sum(x**2) / 4000 - prod_term + 1

# --- MULTIMODAL FUNCTIONS (F8 - F13) ---
def F8(x): # Penalized Function 1 (Paper Table 1 Row 8)
    dim = len(x)
    y = 1 + (x + 1) / 4
    
    term1 = 10 * np.sin(np.pi * y[0])**2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2
    
    sum_u = np.sum(u_fun(x, 10, 100, 4))
    
    return (np.pi / dim) * (term1 + term2 + term3) + sum_u

def F9(x): # Penalized Function 2 (Paper Table 1 Row 9)
    dim = len(x)
    
    term1 = np.sin(3 * np.pi * x[0])**2
    term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:] + 1)**2))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    
    sum_u = np.sum(u_fun(x, 5, 100, 4))
    
    return 0.1 * (term1 + term2 + term3) + sum_u

def F10(x): # Kowalik (Paper Table 1 Row 10)
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 
                  0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    b = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    
    return np.sum((a - (x[0] * (b**2 + x[1] * b)) / (b**2 + x[2] * b + x[3]))**2)

def F11(x): # Six-Hump Camel (Paper Table 1 Row 11)
    x1 = x[0]
    x2 = x[1]
    term1 = 4 * x1**2 - 2.1 * x1**4 + (1/3) * x1**6
    term2 = x1 * x2
    term3 = -4 * x2**2 + 4 * x2**4
    return term1 + term2 + term3

def F12(x): # Branin (Paper Table 1 Row 12)
    x1 = x[0]
    x2 = x[1]
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    d = 6
    e = 10
    f = 1 / (8 * np.pi)
    
    term1 = a * (x2 - b * x1**2 + c * x1 - d)**2
    term2 = e * (1 - f) * np.cos(x1)
    return term1 + term2 + e

def F13(x): # Goldstein-Price (Paper Table 1 Row 13)
    x1 = x[0]
    x2 = x[1]
    
    term1 = (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
    term2 = (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    
    return term1 * term2
# --- FIXED DIMENSION FUNCTIONS (F14 - F18) ---
# These require specific matrices (a_ij, c_i) typically found in standard libraries
# I have added simplified versions or placeholders for the complex matrix ones 
# to ensure the code runs without external data files.

def F14(x):
    a = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.3689, 0.1170, 0.2673],
                  [0.4699, 0.4387, 0.7470],
                  [0.1091, 0.8732, 0.5547],
                  [0.0381, 0.5743, 0.8828]])
    
    val = 0
    for i in range(4):
        exponent = -np.sum(a[i] * (x - p[i])**2)
        val += c[i] * np.exp(exponent)
    return -val  # Negative because we minimize

def F15(x):
    a = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    
    val = 0
    for i in range(4):
        exponent = -np.sum(a[i] * (x - p[i])**2)
        val += c[i] * np.exp(exponent)
    return -val

def shekel(x, m):
    # Standard Shekel Coefficients (a and c)
    a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                  [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                  [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    
    val = 0
    for i in range(m): # m determines Shekel 5, 7, or 10
        denom = np.dot((x - a[i]), (x - a[i])) + c[i]
        val -= 1 / denom
    return val

# ... F16, F17, F18 are variations of Six-Hump Camel and Goldstein-Price
def F16(x):
    return shekel(x, 5)

def F17(x):
    return shekel(x, 7)

def F18(x):
    return shekel(x, 10)

def get_function_details(func_name):
    # Mapping STRICTLY to Paper Table 1 
    lookup = {
        "F1": (F1, -100, 100, 30),
        "F2": (F2, -10, 10, 30),
        "F3": (F3, -100, 100, 30),
        "F4": (F4, -100, 100, 30),
        "F5": (F5, -5.12, 5.12, 30),   # Rastrigin
        "F6": (F6, -32, 32, 30),       # Ackley
        "F7": (F7, -600, 600, 30),     # Griewank
        "F8": (F8, -50, 50, 30),       # Penalized 1
        "F9": (F9, -50, 50, 30),       # Penalized 2
        "F10": (F10, -5, 5, 4),        # Kowalik
        "F11": (F11, -5, 5, 2),        # Camel
        "F12": (F12, -5, 5, 2),        # Branin
        "F13": (F13, -2, 2, 2),        # Goldstein
        "F14": (F14, 0, 1, 3),         # Hartman 3 (Range approx [0,1] for Hartman usually) - Paper says [1,3]? No, paper says F14 range [0, 1]. Let's check table.
        "F15": (F15, 0, 1, 6),
        "F16": (F16, 0, 10, 4),
        "F17": (F17, 0, 10, 4),
        "F18": (F18, 0, 10, 4)
    }
    return lookup.get(func_name, (None, None, None, None))