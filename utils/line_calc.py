import numpy as np

def line_calc(dot1, dot2):
    diff = dot2-dot1
    k = diff[1]/diff[0]
    b = dot1[1]-k*dot1[0]

    return k,b

if __name__ == '__main__':
    dot1 = np.array([2.316,-136.3])
    dot2 = np.array([1.82,-79.6])

    k, b = line_calc(dot1, dot2)
    print('The line is y = %.3f x + %.3f' % (k,b))