import torch
import sys
sys.path.append('..')
sys.path.append('../..')
from pointnet2_utils import matrix_k_min

m = torch.randn(5, 10, 20).cuda(1)
idx = matrix_k_min(0.1, 5, m.cuda(0))
print(idx)
# idx = matrix_k_min(0.1, 5, m.cuda(1))
# print(idx)






