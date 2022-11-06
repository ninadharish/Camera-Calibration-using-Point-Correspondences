import numpy as np


def RQ(M):

    Q, R = np.linalg.qr(np.rot90(M, 3))

    R = np.transpose(np.rot90(R, 2))
    Q = np.rot90(Q)

    return R, Q



def calibrate():

    U1 = [757, 213, 1, 0, 0, 0]
    U2 = [758, 415, 1, 0, 3, 0]
    U3 = [758, 686, 1, 0, 7, 0]
    U4 = [759, 966, 1, 0, 11, 0]
    U5 = [1190, 172, 1, 7, 1, 0]
    U6 = [329, 1041, 1, 0, 11, 7]
    U7 = [1204, 850, 1, 7, 9, 0]
    U8 = [340, 159, 1, 0, 1, 7]

    U = np.array([U1, U2, U3, U4, U5, U6, U7, U8])


    A = []
    for i in range(8):

        A.append([[(U[i, 3]), (U[i, 4]), (U[i, 5]), 1, 0, 0, 0, 0, ((-U[i, 0])*(U[i, 3])), ((-U[i, 0])*(U[i, 4])), ((-U[i, 0])*(U[i, 5])), (-U[i, 0])], [0, 0, 0, 0, (U[i, 3]), (U[i, 4]), (U[i, 5]), 1, ((-U[i, 1])*(U[i, 3])), ((-U[i, 1])*(U[i, 4])), ((-U[i, 1])*(U[i, 5])), (-U[i, 1])]])

    A = np.array(A)
    A = A.reshape(16, 12)

    _, __, Vt = np.linalg.svd(A)

    P = Vt[-1]
    P = np.divide(P, P[-1])

    P = np.reshape(P, (3, 4))

    M = P[:, 0:3]
    
    K, ___ = RQ(M)

    K = np.divide(K, K[-1][-1])

    print("Intrinsic Camera Matrix K = ")
    print(K)



if __name__ == "__main__":

    calibrate()