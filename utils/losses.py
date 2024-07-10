import torch
from torch.nn import functional as F
import numpy as np


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
        z_sum = torch.sum(score[:, i, ...] )
        y_sum = torch.sum(target[:, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target.shape[1]

    return loss

def _get_compactness_cost(y_pred, y_true):

    """
    y_pred: BxHxWxC
    """
    """
    lenth term
    (5,2,384,384)
    """


    # y_pred = tf.one_hot(y_pred, depth=2)
    # print (y_true.shape)
    # print (y_pred.shape)
    y_pred1 = y_pred[:,0 ,...]
    y_pred2 = y_pred[:,1 ,...]
    y_true = y_pred[..., 1]

    x = y_pred1[:,1:,:] - y_pred1[:,:-1,:] # horizontal and vertical directions
    y = y_pred1[:,:,1:] - y_pred1[:,:,:-1]
    x1 = y_pred2[:, 1:, :] - y_pred2[:, :-1, :]
    y1 = y_pred2[:, :, 1:] - y_pred2[:, :, :-1]
    #x1 = y_pred1[:, 1:, :] - y_pred1[:, :-1, :]  # horizontal and vertical directions
    #y1 = y_pred1[:, :, 1:] - y_pred1[:, :, :-1]
    delta_x = x[:,:,1:]**2
    delta_y = y[:,1:,:]**2
    delta_x1 = x1[:, :, 1:] ** 2
    delta_y1 = y1[:, 1:, :] ** 2

    delta_u = torch.abs(delta_x + delta_y)
    delta_u1 = torch.abs(delta_x1 + delta_y1)
    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 0.01
    length = w * torch.sum(torch.sqrt(delta_u + epsilon), [1, 2])
    length1 = w * torch.sum(torch.sqrt(delta_u1 + epsilon), [1, 2])
    #tf.reduce_sum <==>torch.sum
    area = torch.sum(y_pred1, [1,2])
    area1 = torch.sum(y_pred2, [1, 2])

    compactness_loss = torch.sum(length1 ** 2 / (area1 * 4 * 3.1415926))

    return compactness_loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
