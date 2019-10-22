import torch
import numpy as np


def main():
    # matrix multiplication 矩阵点乘
    data = [[1, 2], [3, 4]]
    tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
    # correct method
    print(
        '\nmatrix multiplication (matmul)',
        '\nnumpy: ', np.matmul(data, data),  # [[7, 10], [15, 22]]
        '\ntorch: ', torch.mm(tensor, tensor)  # [[7, 10], [15, 22]]
    )

    # !!!!  下面是错误的方法 !!!!
    data = np.array(data)
    print(
        '\nmatrix multiplication (dot)',
        '\nnumpy: ', data.dot(data),  # [[7, 10], [15, 22]] 在numpy 中可行
        '\ntorch: ', tensor.dot(tensor)  # torch 会转换成 [1,2,3,4].dot([1,2,3,4) = 30.0
    )


if __name__ == '__main__':
    main()