import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import skimage.io as io
import torchvision.transforms as transforms
import config
from tqdm import tqdm
from CAPS.caps_model import CAPSModel


class HPatchDataset(Dataset):
    def __init__(self, imdir):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ])
        self.imfs = []
        for f in os.listdir(imdir):
            scene_dir = os.path.join(imdir, f)
            self.imfs.extend([os.path.join(scene_dir, '{}.ppm').format(ind) for ind in range(1, 7)])

    def __getitem__(self, item):
        imf = self.imfs[item]
        im = io.imread(imf)
        im_tensor = self.transform(im)
        # using sift keypoints
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        kpts = sift.detect(gray)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        coord = torch.from_numpy(kpts).float()
        out = {'im': im_tensor, 'coord': coord, 'imf': imf}
        return out

    def __len__(self):
        return len(self.imfs)


if __name__ == '__main__':
    # example code for extracting features for HPatches dataset, SIFT keypoint is used
    args = config.get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = HPatchDataset(args.extract_img_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    model = CAPSModel(args)

    outdir = args.extract_out_dir
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        for data in tqdm(data_loader):
            im = data['im'].to(device)
            img_path = data['imf'][0]
            coord = data['coord'].to(device)
            feat_c, feat_f = model.extract_features(im, coord)
            desc = torch.cat((feat_c, feat_f), -1).squeeze(0).detach().cpu().numpy()
            kpt = coord.cpu().numpy().squeeze(0)

            out_path = os.path.join(outdir, '{}-{}'.format(os.path.basename(os.path.dirname(img_path)),
                                                           os.path.basename(img_path),
                                                           ))
            with open(out_path + '.caps', 'wb') as output_file:
                np.savez(
                    output_file,
                    keypoints=kpt,
                    scores=[],
                    descriptors=desc
                )

