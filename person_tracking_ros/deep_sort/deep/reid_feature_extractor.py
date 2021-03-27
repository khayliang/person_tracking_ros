import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .osnet_ain import osnet_ain_x1_0

class ReidExtractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.osnet = osnet_ain_x1_0(checkpoint=model_path)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.osnet.to(self.device)
        self.size = (128, 256)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        processed_arr = [self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops]
        im_batch = torch.cat(processed_arr, dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.osnet(im_batch, return_featuremaps=True, fc=True)
            #features = [self.pool2d(feature) for feature in features]
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("./pictures/margo1_cropped.jpg")[:,:,(2,1,0)]
    extr = ReidExtractor("checkpoint/ckpt.t7")
    feature = extr([img])

