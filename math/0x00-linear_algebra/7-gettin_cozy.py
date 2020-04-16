#!/usr/bin/env python3


def cat_matrices2D(mat1, mat2, axis=0):
    newy = []
    newy = mat1.copy()

    # print("1:{} 2:{}".format(mat1, mat2))
    # print("newy = ", newy)

    if (axis == 0):
        for i in mat2:
            newy.append(i)
    if (axis == 1):
        for i in range(len(mat2)):
            newy[i].extend(mat2[i])

    # print("--final", newy, "\n", mat1, "\n", mat2)
    return newy
