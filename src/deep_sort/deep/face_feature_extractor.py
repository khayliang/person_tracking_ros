import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .inception_resnet_v1 import InceptionResnetV1

class FaceExtractor(object):
    def __init__(self, use_cuda=True):
        self.net = InceptionResnetV1(pretrained="vggface2").eval()
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net.to(self.device)
        self.size = (160, 160)
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
        
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0)
        im_batch = im_batch.float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            #features = self.net(im_batch)
            features = self.net(im_batch)
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
