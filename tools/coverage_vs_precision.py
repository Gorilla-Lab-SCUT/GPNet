import numpy as np
from matplotlib import pyplot as plt
import torch


def dist_matrix_torch(x, y):
    x2 = torch.sum(x**2, -1, keepdim=True)
    y2 = torch.sum(y**2, -1, keepdim=True)
    xy = torch.matmul(x, y.transpose(-1, -2))
    matrix = x2 - 2*xy + y2.transpose(-1, -2)
    matrix[matrix<=0] = 1e-10
    return torch.sqrt(matrix)


class coverage_vs_precision(object):
    def __init__(self, gt_data):
        """
        gt_data: annotations. Must consist of success, centers, orientations. 
        """
        self.gripperCenter = gt_data[:, 1:4]
        # print(self.gripperCenter.shape)
        self.gripperOrientation = gt_data[:, 4:]
        self.total_num = gt_data.shape[0]


    @staticmethod
    def to_find(a, b, threshold, dist='Euclidean', gpu=False):
        """
        :param a: n x k.
        :param b: m x k.
        :param threshold: float32.
        :return: a boolean numpy array bool_a with shape [n,], in which bool_a[i]=1 means a[i,...] belongs to b
        in the sense that there exist at least j-th element in b that dist(a[i,...],b[j,...]) is less than threshold. At
        the same time, a boolean numpy array bool_b with shape [m,]. If bool_b[i]=1, it means bool_b[i] is covered.
        Here, dist is Euclidean.
        """
        
        #print(np.min(np.reshape(square_a + square_b - ab,[-1])))
        if dist=='Euclidean':
            # square_a = np.expand_dims(a**2,1)
            # square_b = np.expand_dims(b**2,0)
            # ab = 2*np.multiply(np.expand_dims(a, 1),np.expand_dims(b, 0)) # [n, m]
            # d = np.sqrt(np.maximum(square_a + square_b - ab,0))
            # print(square_a.shape, square_b.shape, ab.shape)
            if not gpu:
                a2 = np.sum(a**2, 1, keepdims=True)
                b2 = np.sum(b**2, 1, keepdims=True)
                ab = 2 * a.dot(b.transpose())
                d = a2 - ab + b2.transpose()
                d = np.sqrt(np.maximum(d, 0))
                d = np.expand_dims(d, -1)
            else:
                x = torch.from_numpy(a).cuda().float()
                y = torch.from_numpy(b).cuda().float()
                d = dist_matrix_torch(x, y)
        elif dist=='Quaternion':
            if not gpu:
                d_ = np.abs(a.dot(np.transpose(b, [1,0])))
                d = 2 * np.arccos(d_)
                d = np.expand_dims(d, -1)
            else:
                x = torch.from_numpy(a).cuda().float()
                y = torch.from_numpy(b).cuda().float()
                d = torch.abs(torch.matmul(x, y.t()))
                d = 2 * torch.acos(d)
        else:
            print('error: please select Euclidean or Quaternion')
        
        # bool = np.all(np.less_equal(d, threshold),-1)
        bool = d < threshold
        # print(bool.shape, d.shape)
        return bool


    def precision_and_recall_at_k_percent(self, predicted_c, predicted_o, k_percent,
                                          threshold_c=5e-3, threshold_o=np.inf, gpu=False):

        k = int(k_percent * predicted_c.shape[0])
        if k == 0:
            k = 1
        # print(k)
        predicted_c, predicted_o = predicted_c[:k, ...], predicted_o[:k, ...] 
        if threshold_o is not np.inf:
            x = (threshold_c, threshold_o)
        else:
            x = threshold_c

        if threshold_o is not np.inf:
            bool_cent = self.to_find(predicted_c, self.gripperCenter, x[0], gpu=gpu)
            bool_quat = self.to_find(predicted_o, self.gripperOrientation, x[1], dist='Quaternion', gpu=gpu) 
            bool_all = bool_cent * bool_quat
            if type(bool_all) is np.ndarray:
                bool_coverage = np.any(bool_all, 0)
                bool_success = np.any(bool_all, 1)
            else:
                bool_coverage = bool_all.any(0).cpu().numpy()
                bool_success = bool_all.any(1).cpu().numpy()
            coverage = np.sum(bool_coverage) / self.total_num
            presicion = np.mean(bool_success)
            return presicion, coverage, k
        else:
            assert False, 'error!!!'


