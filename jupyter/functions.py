import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
import sys
sys.path.append('../')
from dataloader import megadepth
import torch.utils.data
from CAPS.caps_model import CAPSModel
import cv2


class Visualization(object):
    def __init__(self, args):
        dataset = megadepth.MegaDepth(args)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
        self.model = CAPSModel(args)
        self.loader_iter = iter(self.dataloader)

    def random_sample(self):
        self.sample = next(self.loader_iter)

    def plot_img_pair(self, with_std=False, with_epipline=False):
        self.coords = []
        self.colors = []
        self.with_std = with_std
        self.with_epipline = with_epipline
        im1 = self.sample['im1_ori']
        im2 = self.sample['im2_ori']
        self.h, self.w = im1.shape[1], im1.shape[2]
        im1 = im1.squeeze().cpu().numpy()
        im2 = im2.squeeze().cpu().numpy()
        blank = np.ones((self.h, 5, 3)) * 255
        out = np.concatenate((im1, blank, im2), 1).astype(np.uint8)
        self.fig = plt.figure(figsize=(12, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(out)
        self.ax.axis('off')
        plt.tight_layout()
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        color = tuple(np.random.rand(3).tolist())
        coord = [event.xdata, event.ydata]
        self.coord = coord
        self.color = color
        self.coords.append(coord)
        self.colors.append(color)
        self.ax.scatter(event.xdata, event.ydata, c=color)
        self.find_correspondence()
        self.plot_correspondence()

    def find_correspondence(self):
        data_in = self.sample
        data_in['coord1'] = torch.from_numpy(np.array(self.coord)).float().cuda().unsqueeze(0).unsqueeze(0)
        data_in['coord2'] = data_in['coord1']
        self.model.set_input(data_in)
        coord2_e, std = self.model.test()
        self.correspondence = coord2_e.squeeze().cpu().numpy()
        self.std = std.squeeze().cpu().numpy()

    def plot_correspondence(self):
        point1 = self.coord
        point2 = self.correspondence
        point2[0] += self.w + 5
        self.ax.scatter(point2[0], point2[1], color=self.color)
        if self.with_std:
            circle = plt.Circle((point2[0], point2[1]), radius=100 * self.std, fill=False, color=self.color)
            self.ax.add_patch(circle)

        if self.with_epipline:
            line2 = cv2.computeCorrespondEpilines(np.array(point1).reshape(-1, 1, 2), 1,
                                                   self.sample['F'].squeeze().cpu().numpy())
            line2 = np.array(line2).squeeze()
            intersection = np.array([[0, -line2[2]/line2[1]],
                                     [-line2[2]/line2[0], 0],
                                     [self.w-1, -(line2[2]+line2[0]*(self.w-1))/line2[1]],
                                     [-(line2[1]*(self.h-1)+line2[2])/line2[0], self.h-1]])
            valid = (intersection[:, 0] >= 0) & (intersection[:, 0] <= self.w-1) & \
                    (intersection[:, 1] >= 0) & (intersection[:, 1] <= self.h-1)
            if np.sum(valid) == 2:
                intersection = intersection[valid].astype(int)
                x0, y0 = intersection[0]
                x1, y1 = intersection[1]
                l = mlines.Line2D([x0+self.w + 5, x1+self.w + 5], [y0, y1], color=self.color)
                self.ax.add_line(l)

        plt.show()









