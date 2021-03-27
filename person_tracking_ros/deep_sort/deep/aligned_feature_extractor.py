import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .reid_feature_extractor import ReidExtractor
from .osnet_ain import osnet_ain_x1_0

class AlignedExtractor(ReidExtractor):

    
    def __init__(self, model_path, use_cuda=True):
        super().__init__(model_path, use_cuda=True)

    
    def pool2d(self, tensor, type= 'max'):
        sz = tensor.size()

        if type == 'max':
            x = torch.nn.functional.max_pool2d(tensor, kernel_size=(int(sz[1]/8), sz[2]))
        if type == 'mean':
            x = torch.nn.functional.mean_pool2d(tensor, kernel_size=(int(sz[1]/8), sz[2]))

        x = x.cpu().data.numpy()
        x = np.transpose(x,(2,1,0))[0]

        return x


    def __call__(self, im_crops):
        im_batch = super()._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.osnet(im_batch, return_featuremaps=True, fc=False)
            features = [self.pool2d(feature) for feature in features]
        return features


if __name__ == '__main__':
    img = cv2.imread("./pictures/margo1_cropped.jpg")[:,:,(2,1,0)]
    extr = ReidExtractor("checkpoint/ckpt.t7")
    feature = extr([img])

