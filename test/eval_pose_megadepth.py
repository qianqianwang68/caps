import numpy as np
import cv2
import sys
import os
sys.path.append('../')
import scipy
import utils
from skimage import io
from pathos.multiprocessing import ProcessingPool as Pool
import csv
from datetime import datetime
import collections
import struct


class MegaDepthPose(object):
    def __init__(self, pose_args, mode):
        root = '/phoenix/S7/qw246/mega-sub/test-pose/'
        self.root = os.path.join(root, mode)
        self.img_folder = os.path.join(os.path.dirname(root), 'megaDepth', 'test')  # your image folder path
        self.images = self.read_img_cam()
        self.pose_args = pose_args
        self.method = pose_args.method
        self.kp_desc_path = os.path.join(root, 'output', 'desc', self.method)  # your kpt and descriptor path
        self.phase = 'test'

    def read_img_cam(self):
        images = {}
        Image = collections.namedtuple(
            "Image", ["name", "w", "h", "fx", "fy", "cx", "cy", "rvec", "tvec"])
        # TODO: now only dense/ is used, try to include dense1/...
        img_cam_txt_path = os.path.join(self.root, 'img_cam.txt')
        with open(img_cam_txt_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    folder_id = elems[0]
                    image_name = elems[1]
                    img_path = os.path.join(self.img_folder, folder_id, 'dense', 'aligned', 'images', image_name)
                    w, h = int(elems[2]), int(elems[3])
                    fx, fy = float(elems[4]), float(elems[5])
                    cx, cy = float(elems[6]), float(elems[7])
                    R = np.array(elems[8:17])
                    T = np.array(elems[17:20])
                    images[img_path] = Image(
                        name=image_name, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, rvec=R, tvec=T
                    )
        return images

    def read_pairs(self):
        imf1s, imf2s, pairfs = [], [], []
        pairf = os.path.join(self.root, 'pairs.txt')
        f = open(pairf, 'r')
        for line in f:
            folder_id, imn1, imn2 = line.strip().split(' ')
            # the paths below should be your actual image paths
            imf1s.append(os.path.join(self.img_folder, folder_id, 'dense', 'aligned', 'images', imn1))
            imf2s.append(os.path.join(self.img_folder, folder_id, 'dense', 'aligned', 'images', imn2))
            pairfs.append(os.path.join(self.root, 'pairs', '{}-{}-{}.txt'.format(folder_id, imn1, imn2)))
        return imf1s, imf2s, pairfs

    def load_kp_desc(self, imf):
        # you need to precompute the keypoints and descriptors
        kp_desc_fn = os.path.join(self.kp_desc_path, '{}-{}.npz'.format(imf.split('/')[-5], os.path.basename(imf)))
        kp_desc_f = np.load(kp_desc_fn)
        kp = kp_desc_f['keypoints'][:, :2]
        desc = kp_desc_f['descriptors']
        return kp, desc

    def compose_intrinsic_extrinsic(self, imf):
        im_meta = self.images[imf]
        intrinsic = np.array([[im_meta.fx, 0, im_meta.cx],
                               [0, im_meta.fy, im_meta.cy],
                               [0, 0, 1]])
        R = im_meta.rvec.reshape(3, 3)
        t = im_meta.tvec
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        return intrinsic, extrinsic

    def get_pose_error(self, kp1, kp2, desc1, desc2, intrinsic1, intrinsic2, pose, args, imf1, imf2):
        use_ratio = args.use_ratio
        use_dist = args.use_dist
        use_prob = args.use_prob
        ratio = args.ratio
        dist_th = args.dist_th
        opencv_matcher = args.opencv_matcher
        prob_th = args.prob_th

        if args.unique:
            kp1, unique_idx = np.unique(kp1, return_index=True, axis=0)
            desc1 = desc1[unique_idx]
            kp2, unique_idx = np.unique(kp2, return_index=True, axis=0)
            desc2 = desc2[unique_idx]

        if opencv_matcher:
            if use_ratio:
                good = []
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(desc1, desc2, k=2)
                for m, n in matches:
                    if m.distance < ratio * n.distance:
                        good.append(m)
                if len(good) < 50:
                    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
                    good = [m[0] for m in matches[:50]]

            elif use_dist:
                bf = cv2.BFMatcher(crossCheck=bool(args.cross_check))
                matches = bf.match(desc1, desc2)
                good = []
                for m in matches:
                    if m.distance < dist_th:
                        good.append(m)
                if len(good) < 50:
                    matches = sorted(matches, key=lambda x: x.distance)
                    good = [m for m in matches[:50]]
            else:
                bf = cv2.BFMatcher(crossCheck=bool(args.cross_check))
                good = bf.match(desc1, desc2)

                bf = cv2.BFMatcher()
                matches = bf.match(desc1, desc2)
                if len(good) < 50:
                    matches = sorted(matches, key=lambda x: x.distance)
                    good = [m for m in matches[:50]]

            queryIdx = [m.queryIdx for m in good]
            trainIdx = [m.trainIdx for m in good]
            kp1, kp2 = kp1[queryIdx], kp2[trainIdx]
            kp1 = np.ascontiguousarray(kp1, dtype=np.float32)
            kp2 = np.ascontiguousarray(kp2, dtype=np.float32)

        else:
            if use_prob:
                sim = desc1.dot(desc2.T) / 2
                nn12 = np.argmax(sim, axis=1)
                sim12 = np.max(sim, axis=1)
                nn21 = np.argmax(sim, axis=0)
                ids1 = np.arange(0, sim.shape[0])
                mask = (ids1 == nn21[nn12])
                from scipy.special import softmax
                probs_mask = np.max(softmax(sim, axis=1), axis=1) > prob_th
                mask *= probs_mask

                good = np.stack([ids1[mask], nn12[mask]], axis=1)
                if len(good) < 50:
                    idx_sim_score = np.argsort(-sim12)[:50]
                    good = np.stack([ids1[idx_sim_score], nn12[idx_sim_score]], axis=1)

            else:
                from sklearn.metrics import pairwise_distances
                dist = pairwise_distances(desc1, desc2)
                nn12 = np.argmin(dist, axis=1)
                dist12 = np.min(dist, axis=1)
                nn21 = np.argmin(dist, axis=0)
                ids1 = np.arange(0, dist.shape[0])
                mask = (ids1 == nn21[nn12])
                if use_ratio:
                    dist_sorted = np.sort(dist, axis=1)
                    mask_ratio = dist_sorted[:, 0] / dist_sorted[:, 1]
                    mask *= mask_ratio < ratio

                good = np.stack([ids1[mask], nn12[mask]], axis=1)
                if len(good) < 50:
                    idx_sim_score = np.argsort(dist12)[:50]
                    good = np.stack([ids1[idx_sim_score], nn12[idx_sim_score]], axis=1)

            kp1 = np.ascontiguousarray(kp1[good[:, 0]], dtype=np.float32)
            kp2 = np.ascontiguousarray(kp2[good[:, 1]], dtype=np.float32)

        intrinsic_mean = (intrinsic1 + intrinsic2) / 2.
        focal = (intrinsic_mean[0, 0] + intrinsic_mean[1, 1]) / 2.
        pp = tuple(intrinsic_mean[0:2, 2])

        E, mask = cv2.findEssentialMat(kp1, kp2, focal=focal, pp=pp, method=cv2.RANSAC)
        try:
            R1, R2, t = cv2.decomposeEssentialMat(E)
        except:
            print(imf1, imf2)
            print(E)

        R_gt, t_gt = pose[:3, :3], pose[:3, 3]
        theta_1 = np.arccos(np.clip((np.trace(R1.T.dot(R_gt)) - 1) / 2, -1, 1))
        theta_2 = np.arccos(np.clip((np.trace(R2.T.dot(R_gt)) - 1) / 2, -1, 1))
        theta = min(theta_1, theta_2) * 180 / np.pi
        t = np.squeeze(t)
        tran_cos = np.inner(t, t_gt) / (np.linalg.norm(t_gt) * np.linalg.norm(t))
        tran = np.arccos(tran_cos) * 180 / np.pi
        tran = min(tran, 180 - tran)
        print(len(good), theta, tran)
        return theta, tran, kp1, kp2, mask

    def visMatch(self, imf1, imf2, pt1, pt2, mask, text, R_error, T_error, savepath):
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))

        im1 = io.imread(imf1)
        im2 = io.imread(imf2)
        # resize rgb image to the same size as depth

        # pt1 = [cv2.KeyPoint(p[0], p[1], 1) for p in pt1]
        # pt2 = [cv2.KeyPoint(p[0], p[1], 1) for p in pt2]
        # out = np.array([])
        # out = cv2.drawMatches(im1, pt1, im2, pt2, [cv2.DMatch(i, i, 0) for i in range(len(pt1))],
        # out, matchesMask=mask.ravel().tolist())
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale = 1
        # fontColor = (255, 0, 0)
        # lineType = 2
        #
        # cv2.putText(out, '{} angular error: R {:.3f}, T {:.3f}'.format(text, R_error, T_error),
        #             (int(im1.shape[1] / 2 - 80), 100),
        #             font, fontScale, fontColor, lineType)
        pt1 = pt1[mask.ravel().astype(bool)]
        pt2 = pt2[mask.ravel().astype(bool)]
        out = utils.drawlinesMatch(im1, im2, pt1, pt2)
        io.imsave(savepath, out)

    def pose_parallel(self, imf1, imf2, vis=False):
        kp1, desc1 = self.load_kp_desc(imf1)
        kp2, desc2 = self.load_kp_desc(imf2)

        intrinsic1, extrinsic1 = self.compose_intrinsic_extrinsic(imf1)
        intrinsic2, extrinsic2 = self.compose_intrinsic_extrinsic(imf2)

        pose = extrinsic2.dot(np.linalg.inv(extrinsic1))
        R_error, T_error, kp1, kp2, mask = self.get_pose_error(kp1, kp2, desc1, desc2, intrinsic1, intrinsic2, pose,
                                                               self.pose_args, imf1, imf2)
        if vis == 1:
            savef = '{}/{}/{}/{}_{}.png'.format('vis', pose_args.method, pose_args.mode,
                                                os.path.basename(imf1), os.path.basename(imf2))
            if not os.path.exists(os.path.dirname(savef)):
                os.makedirs(os.path.dirname(savef))
            self.visMatch(imf1, imf2, kp1, kp2, mask, '', R_error, T_error, savef)
        return R_error, T_error

    def run(self):
        outf = os.path.join(os.path.dirname(self.root), 'result', '{}.csv'.format(self.method))
        if not os.path.exists(os.path.dirname(outf)):
            os.makedirs(os.path.dirname(outf))

        imf1s, imf2s, pairfs = self.read_pairs()
        res = Pool().map(self.pose_parallel, imf1s, imf2s, [pose_args.vis]*len(imf1s))
        R_errors, T_errors = np.array(res)[:, 0], np.array(res)[:, 1]
        R_5_accuracy = np.mean(R_errors < 5)
        T_5_accuracy = np.mean(T_errors < 5)
        R_10_accuracy = np.mean(R_errors < 10)
        T_10_accuracy = np.mean(T_errors < 10)
        R_median = np.median(R_errors)
        T_median = np.median(T_errors)
        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print('R_5_accuracy: {}, T_5_accuracy: {}, R_10_accuracy: {}, T_10_accuracy: {}, R_median: {}, T_median: {}'
              .format(np.round(R_5_accuracy, 4), np.round(T_5_accuracy, 4),
                      np.round(R_10_accuracy, 4), np.round(T_10_accuracy, 4),
                      np.round(R_median, 3), np.round(T_median, 3)))

        with open(outf, 'a') as csvfile:
            fieldnames = ['Date', 'mode', 'R_5_accuracy', 'T_5_accuracy', 'R_10_accuracy', 'T_10_accuracy',
                          'R_median', 'T_median']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if os.path.getsize(outf) == 0:
                writer.writeheader()
            writer.writerow({'Date': time_now, 'mode': self.pose_args.mode,
                             'R_5_accuracy': np.round(R_5_accuracy, 4), 'T_5_accuracy': np.round(T_5_accuracy, 4),
                             'R_10_accuracy': np.round(R_10_accuracy, 4), 'T_10_accuracy': np.round(T_10_accuracy, 4),
                             'R_median': np.round(R_median, 3), 'T_median': np.round(T_median, 3)})


if __name__ == '__main__':
    ## The defaults args below are what we use in the paper.
    # We set only cross check and unique to be true for our method.
    import argparse
    parser = argparse.ArgumentParser(description='arguments for estimating pose')
    parser.add_argument('--method', type=str, help='the method to test on')
    parser.add_argument('--use_ratio', type=int, default=0, help='if use ratio test')
    parser.add_argument('--ratio', type=float, default=0.8, help='the ratio for ratio test')
    parser.add_argument('--use_prob', type=int, default=0, help='if use probability test')
    parser.add_argument('--prob_th', type=float, default=0.5, help='use the probability as thresholding')
    parser.add_argument('--use_dist', type=int, default=0, help='if use distance as the filtering method')
    parser.add_argument('--dist_th', type=float, default=7, help='the threshold for distance')
    parser.add_argument('--cross_check', type=int, default=1, help='if use cross check')
    parser.add_argument('--opencv_matcher', type=int, default=0, help='if use opencv matcher')
    parser.add_argument('--unique', type=int, default=1, help='remove duplicate keypoints that have the same xy location')
    parser.add_argument('--vis', type=int, default=0, help='if visualize the matching results')
    parser.add_argument('--mode', type=str, default='easy', help='which dataset to test on: easy, hard, tough')

    pose_args = parser.parse_args()

    tester = MegaDepthPose(pose_args, pose_args.mode)
    print(pose_args.method, pose_args.mode)
    tester.run()

