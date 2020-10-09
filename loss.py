import torch
import numpy as np

def angle_loss(theta, theta_gt, mask_gt, epsilon=1e-2):
    '''
    theta: (BS, 1)
    theta: (BS, n, 1)
    mask_gt: (BS, n, 1)
    '''
    # print(theta.size(), theta_gt.size(), mask_gt.size())
    M = torch.sum(mask_gt, 1).float()
    maximum, minimum = (1 - epsilon * (M + 1) / M), epsilon / M
    diff = theta.unsqueeze(1) - theta_gt
    norm_with_gt = diff**2 / (1e-8 + mask_gt.float())
    main_loss, index = torch.min(norm_with_gt, 1) #(BS, 1)
    # main_loss = torch.gather(torch.squeeze(norm_with_gt), 1, index)
    residual_loss = mask_gt.float() * diff**2
    loss1 = torch.sum(maximum * main_loss, 1)
    loss2 = torch.sum(minimum.unsqueeze(1)*residual_loss, dim=1).sum(1)
    loss = loss1 + loss2
    return loss
