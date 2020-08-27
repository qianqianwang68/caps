import torch
import torch.nn as nn


class CtoFCriterion(nn.Module):
    def __init__(self, args):
        super(CtoFCriterion, self).__init__()
        self.args = args
        self.w_ec = args.w_epipolar_coarse
        self.w_ef = args.w_epipolar_fine
        self.w_cc = args.w_cycle_coarse
        self.w_cf = args.w_cycle_coarse
        self.w_std = args.w_std

    def homogenize(self, coord):
        coord = torch.cat((coord, torch.ones_like(coord[:, :, [0]])), -1)
        return coord

    def set_weight(self, std, mask=None, regularizer=0.0):
        if self.args.std:
            inverse_std = 1. / torch.clamp(std+regularizer, min=1e-10)
            weight = inverse_std / torch.mean(inverse_std)
            weight = weight.detach()  # Bxn
        else:
            weight = torch.ones_like(std)

        if mask is not None:
            weight *= mask.float()
            weight /= (torch.mean(weight) + 1e-8)
        return weight

    def epipolar_cost(self, coord1, coord2, fmatrix):
        coord1_h = self.homogenize(coord1).transpose(1, 2)
        coord2_h = self.homogenize(coord2).transpose(1, 2)
        epipolar_line = fmatrix.bmm(coord1_h)  # Bx3xn
        epipolar_line_ = epipolar_line / torch.clamp(torch.norm(epipolar_line[:, :2, :], dim=1, keepdim=True), min=1e-8)
        essential_cost = torch.abs(torch.sum(coord2_h * epipolar_line_, dim=1))  # Bxn
        return essential_cost

    def epipolar_loss(self, coord1, coord2, fmatrix, weight):
        essential_cost = self.epipolar_cost(coord1, coord2, fmatrix)
        loss = torch.mean(weight * essential_cost)
        return loss

    def cycle_consistency_loss(self, coord1, coord1_loop, weight, th=40):
        '''
        compute the cycle consistency loss
        :param coord1: [batch_size, n_pts, 2]
        :param coord1_loop: the predicted location  [batch_size, n_pts, 2]
        :param weight: the weight [batch_size, n_pts]
        :param th: the threshold, only consider distances under this threshold
        :return: the cycle consistency loss value
        '''
        distance = torch.norm(coord1 - coord1_loop, dim=-1)
        distance_ = torch.zeros_like(distance)
        distance_[distance < th] = distance[distance < th]
        loss = torch.mean(weight * distance_)
        return loss

    def forward(self, coord1, data, fmatrix, pose, im_size):
        coord2_ec = data['coord2_ec']
        coord2_ef = data['coord2_ef']
        coord1_lc = data['coord1_lc']
        coord1_lf = data['coord1_lf']
        std_c = data['std_c']
        std_f = data['std_f']
        std_lc = data['std_lc']
        std_lf = data['std_lf']
        shorter_edge, longer_edge = min(im_size), max(im_size)

        epipolar_cost_c = self.epipolar_cost(coord1, coord2_ec, fmatrix)
        # only add fine level loss if the coarse level prediction is close enough to gt epipolar line
        mask_ctof = (epipolar_cost_c < (shorter_edge * self.args.window_size))
        # only add cycle consistency loss if the coarse level prediction is close enough to gt epipolar line
        mask_epip_c = (epipolar_cost_c < (shorter_edge * self.args.th_epipolar))
        mask_cycle_c = (epipolar_cost_c < (shorter_edge * self.args.th_cycle))

        epipolar_cost_f = self.epipolar_cost(coord1, coord2_ef, fmatrix)
        # only add cycle consistency loss if the fine level prediction is close enough to gt epipolar line
        mask_epip_f = (epipolar_cost_f < (shorter_edge * self.args.th_epipolar))
        mask_cycle_f = (epipolar_cost_f < shorter_edge * self.args.th_cycle)

        weight_c = self.set_weight(std_c, mask=mask_epip_c)
        weight_f = self.set_weight(std_f, mask=mask_epip_f*mask_ctof)

        eloss_c = torch.mean(epipolar_cost_c * weight_c) / longer_edge
        eloss_f = torch.mean(epipolar_cost_f * weight_f) / longer_edge

        weight_cycle_c = self.set_weight(std_c * std_lc, mask=mask_cycle_c)
        weight_cycle_f = self.set_weight(std_f * std_lf, mask=mask_cycle_f)

        closs_c = self.cycle_consistency_loss(coord1, coord1_lc, weight_cycle_c) / longer_edge
        closs_f = self.cycle_consistency_loss(coord1, coord1_lf, weight_cycle_f) / longer_edge

        loss = self.w_ec * eloss_c + self.w_ef * eloss_f + self.w_cc * closs_c + self.w_cf * closs_f

        std_loss = torch.mean(std_c) + torch.mean(std_f)
        loss += self.w_std * std_loss

        return loss, eloss_c, eloss_f, closs_c, closs_f, std_loss


