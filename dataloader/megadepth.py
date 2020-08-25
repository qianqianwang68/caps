import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import skimage.io as io
import torchvision.transforms as transforms
import utils
import collections
from tqdm import tqdm
import dataloader.data_utils as data_utils


rand = np.random.RandomState(234)


class MegaDepthLoader():
    def __init__(self, args):
        self.args = args
        self.dataset = MegaDepth(args)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=args.workers, collate_fn=self.my_collate)

    def my_collate(self, batch):
        ''' Puts each data field into a tensor with outer dimension batch size '''
        batch = list(filter(lambda b: b is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'MegaDepthLoader'

    def __len__(self):
        return len(self.dataset)


class MegaDepth(Dataset):
    def __init__(self, args):
        self.args = args
        if args.phase == 'train':
            # augment during training
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ColorJitter
                                                 (brightness=1, contrast=1, saturation=1, hue=0.4),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                      std=(0.229, 0.224, 0.225)),
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                      std=(0.229, 0.224, 0.225)),
                                                 ])
        self.phase = args.phase
        self.root = os.path.join(args.datadir, self.phase)
        self.images = self.read_img_cam()
        self.imf1s, self.imf2s = self.read_pairs()
        print('total number of image pairs loaded: {}'.format(len(self.imf1s)))
        # shuffle data
        index = np.arange(len(self.imf1s))
        rand.shuffle(index)
        self.imf1s = list(np.array(self.imf1s)[index])
        self.imf2s = list(np.array(self.imf2s)[index])

    def read_img_cam(self):
        images = {}
        Image = collections.namedtuple(
            "Image", ["name", "w", "h", "fx", "fy", "cx", "cy", "rvec", "tvec"])
        for scene_id in os.listdir(self.root):
            densefs = [f for f in os.listdir(os.path.join(self.root, scene_id))
                       if 'dense' in f and os.path.isdir(os.path.join(self.root, scene_id, f))]
            for densef in densefs:
                folder = os.path.join(self.root, scene_id, densef, 'aligned')
                img_cam_txt_path = os.path.join(folder, 'img_cam.txt')
                with open(img_cam_txt_path, "r") as fid:
                    while True:
                        line = fid.readline()
                        if not line:
                            break
                        line = line.strip()
                        if len(line) > 0 and line[0] != "#":
                            elems = line.split()
                            image_name = elems[0]
                            img_path = os.path.join(folder, 'images', image_name)
                            w, h = int(elems[1]), int(elems[2])
                            fx, fy = float(elems[3]), float(elems[4])
                            cx, cy = float(elems[5]), float(elems[6])
                            R = np.array(elems[7:16])
                            T = np.array(elems[16:19])
                            images[img_path] = Image(
                                name=image_name, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, rvec=R, tvec=T
                            )
        return images

    def read_pairs(self):
        imf1s, imf2s = [], []
        print('reading image pairs from {}...'.format(self.root))
        for scene_id in tqdm(os.listdir(self.root), desc='# loading data from scene folders'):
            densefs = [f for f in os.listdir(os.path.join(self.root, scene_id))
                       if 'dense' in f and os.path.isdir(os.path.join(self.root, scene_id, f))]
            for densef in densefs:
                imf1s_ = []
                imf2s_ = []
                folder = os.path.join(self.root, scene_id, densef, 'aligned')
                pairf = os.path.join(folder, 'pairs.txt')

                if os.path.exists(pairf):
                    f = open(pairf, 'r')
                    for line in f:
                        imf1, imf2 = line.strip().split(' ')
                        imf1s_.append(os.path.join(folder, 'images', imf1))
                        imf2s_.append(os.path.join(folder, 'images', imf2))

                # make # image pairs per scene more balanced
                if len(imf1s_) > 5000:
                    index = np.arange(len(imf1s_))
                    rand.shuffle(index)
                    imf1s_ = list(np.array(imf1s_)[index[:5000]])
                    imf2s_ = list(np.array(imf2s_)[index[:5000]])

                imf1s.extend(imf1s_)
                imf2s.extend(imf2s_)

        return imf1s, imf2s

    @staticmethod
    def get_intrinsics(im_meta):
        return np.array([[im_meta.fx, 0, im_meta.cx],
                         [0, im_meta.fy, im_meta.cy],
                         [0, 0, 1]])

    @staticmethod
    def get_extrinsics(im_meta):
        R = im_meta.rvec.reshape(3, 3)
        t = im_meta.tvec
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        return extrinsic

    def __getitem__(self, item):
        imf1 = self.imf1s[item]
        imf2 = self.imf2s[item]
        im1_meta = self.images[imf1]
        im2_meta = self.images[imf2]
        im1 = io.imread(imf1)
        im2 = io.imread(imf2)
        h, w = im1.shape[:2]

        intrinsic1 = self.get_intrinsics(im1_meta)
        intrinsic2 = self.get_intrinsics(im2_meta)

        extrinsic1 = self.get_extrinsics(im1_meta)
        extrinsic2 = self.get_extrinsics(im2_meta)

        relative = extrinsic2.dot(np.linalg.inv(extrinsic1))
        R = relative[:3, :3]
        # remove pairs that have a relative rotation angle larger than 80 degrees
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
        if theta > 80 and self.phase == 'train':
            return None

        T = relative[:3, 3]
        tx = data_utils.skew(T)
        E_gt = np.dot(tx, R)
        F_gt = np.linalg.inv(intrinsic2).T.dot(E_gt).dot(np.linalg.inv(intrinsic1))

        # generate candidate query points
        coord1 = data_utils.generate_query_kpts(im1, self.args.train_kp, 10*self.args.num_pts, h, w)

        # if no keypoints are detected
        if len(coord1) == 0:
            return None

        # prune query keypoints that are not likely to have correspondence in the other image
        if self.args.prune_kp:
            ind_intersect = data_utils.prune_kpts(coord1, F_gt, im2.shape[:2], intrinsic1, intrinsic2,
                                                  relative, d_min=4, d_max=400)
            if np.sum(ind_intersect) == 0:
                return None
            coord1 = coord1[ind_intersect]

        coord1 = utils.random_choice(coord1, self.args.num_pts)
        coord1 = torch.from_numpy(coord1).float()

        im1_ori, im2_ori = torch.from_numpy(im1), torch.from_numpy(im2)

        F_gt = torch.from_numpy(F_gt).float() / (F_gt[-1, -1] + 1e-10)
        intrinsic1 = torch.from_numpy(intrinsic1).float()
        intrinsic2 = torch.from_numpy(intrinsic2).float()
        pose = torch.from_numpy(relative[:3, :]).float()
        im1_tensor = self.transform(im1)
        im2_tensor = self.transform(im2)

        out = {'im1': im1_tensor,
               'im2': im2_tensor,
               'im1_ori': im1_ori,
               'im2_ori': im2_ori,
               'pose': pose,
               'F': F_gt,
               'intrinsic1': intrinsic1,
               'intrinsic2': intrinsic2,
               'coord1': coord1}

        return out

    def __len__(self):
        return len(self.imf1s)
