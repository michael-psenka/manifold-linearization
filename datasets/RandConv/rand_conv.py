'''
Random multi-scale convolutions
'''

import numpy as np
import random
import cv2

def randConv(X, K):
    k = random.choice(K)
    kernel = np.random.normal(scale=1/(3 * k * k), size=(k, k))
    Xc = cv2.filter2D(X, -1, kernel)
    alpha = np.random.uniform()
    output = alpha * Xc + (1 - alpha) * X

    return output


X = np.random.normal(size=(30, 15))
K = [1, 3, 5, 7, 9 ]
out = randConv(X, K)
print(out.shape)
