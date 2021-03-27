import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .face_feature_extractor import FaceExtractor
from .reid_feature_extractor import ReidExtractor

class FeatureExtractor(object):
    def __init__(self, use_cuda=True):
        self.face_extractor = FaceExtractor()
        self.reid_extractor = ReidExtractor()

    def _preprocess(self, im_crops_tuple):
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
        
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0)
        im_batch = im_batch.float()
        #print(im_batch.shape)
        return im_batch

    
    def __call__(self, im_crops_tuple):
        unzip_tuples = [list(t) for t in zip(*tuples_list)]
        im_crops_face = np.array(unzip_tuples[1])
        im_crops_body = np.array(unzip_tuples[1])
        
        features_body = ReidExtractor(im_crops_body)
        features_face = FaceExtractor(im_crops_face)

        zipped_features = zip()

        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("./pictures/margo2_cropped.jpg")[:,:,(2,1,0)]
    img2 = cv2.imread("./pictures/margo1_cropped.jpg")[:,:,(2,1,0)]
    img3 = cv2.imread("./pictures/tom1_cropped.jpg")[:,:,(2,1,0)]

    extr = FaceExtractor()
    feature = extr([img, img2, img3])
    #print(feature[0].shape)
    a = np.asarray(feature) / np.linalg.norm(feature, axis=1, keepdims=True)
    b = np.asarray(feature) / np.linalg.norm(feature, axis=1, keepdims=True)
    cos_dist= 1. - np.dot(a, b.T)
    print(cos_dist)
