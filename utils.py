import numpy as np
import scipy
import cv2


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def evaluate_pose(E, P):
    R_gt = P[:3, :3]
    t_gt = P[:3, 3]
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = t.squeeze()
    theta_1 = np.linalg.norm(scipy.linalg.logm(R1.T.dot(R_gt)), 'fro') / np.sqrt(2)
    theta_2 = np.linalg.norm(scipy.linalg.logm(R2.T.dot(R_gt)), 'fro') / np.sqrt(2)
    theta = min(theta_1, theta_2) * 180 / np.pi
    tran_cos = np.inner(t, t_gt) / (np.linalg.norm(t_gt) * np.linalg.norm(t))
    tran = np.arccos(tran_cos) * 180 / np.pi
    return theta, tran


def average_precision(labels, logits):
    '''
    inputs: label: num_examples x num_pts
            logits: num_examples x num_pts
    :return: average precision
    '''
    from sklearn.metrics import average_precision_score
    sum_ap = 0
    count = 0
    for label, logit in zip(labels, logits):
        if np.sum(label) == 0:
            continue
        ap = average_precision_score(label, logit)
        sum_ap += ap
        count += 1

    map = sum_ap/count if count != 0 else 0
    return map


def homogenize(kp):
    '''
    turn into homogeneous coordinates
    :param kp: n*2 coordinates
    :return: n*3 coordinates where the last channel is 1
    '''
    ones = np.ones_like(kp[:, 0:1])
    return np.concatenate((kp, ones), 1)


def random_choice(array, size):
    rand = np.random.RandomState(1234)
    num_data = len(array)
    if num_data > size:
        idx = rand.choice(num_data, size, replace=False)
    else:
        idx = rand.choice(num_data, size, replace=True)
    return array[idx]


def drawlines(img1, img2, lines, pts1, pts2, color=None, thickness=-1):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    color_ = color
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        if r[1] == 0:
            continue
        if color_ is None:
            color = tuple(np.random.randint(0, 255, 3).tolist())
        else:
            color = color_
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, thickness)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, thickness)
    return img1, img2


def to_jet(input, type='tensor', mode='HW1'):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap('jet')

    if type == 'tensor':
        input = input.detach().cpu().numpy()

    if mode == '1HW':
        input = input.transpose(1, 2, 0)
    elif mode == 'B1HW':
        input = input.transpose(0, 2, 3, 1)
    elif mode == 'HW':
        input = input[..., np.newaxis]  # hxwx1

    if input.ndim == 3:
        out = cm(input[:, :, 0])[:, :, :3]
    else:
        out = np.zeros_like(input).repeat(3, axis=-1)
        for i, data in enumerate(input):
            out[i] = cm(input[:, :, 0])[:, :, :3]
    return out


def drawlinesMatch(img1, img2, pts1, pts2, concat_row=True):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    interval = 5
    if concat_row:
        out = 255 * np.ones((max([rows1, rows2]), cols1 + cols2+interval, 3), dtype='uint8')
        out[:rows2, cols1+interval:cols1+cols2+interval, :] = img2
        pts2[:, 0] += cols1 + interval
    else:
        out = 255 * np.ones((rows1 + rows2 + interval, max(cols1, cols2), 3), dtype='uint8')
        out[rows1+interval:rows1+rows2+interval, :cols2] = img2
        pts2[:, 1] += rows1 + interval

    # Place the first image to the left
    out[:rows1, :cols1, :] = img1
    thickness = 3
    radius = 5

    for pt1, pt2 in zip(pts1, pts2):
        cv2.circle(out, (int(pt1[0]), int(pt1[1])), radius, tuple(np.array([255, 0, 0]).tolist()), -1, cv2.LINE_AA)
        cv2.circle(out, (int(pt2[0]), int(pt2[1])), radius, tuple(np.array([255, 0, 0]).tolist()), -1, cv2.LINE_AA)
        cv2.line(out, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color=(0, 255, 0),
                 lineType=cv2.LINE_AA, thickness=thickness)
    return out