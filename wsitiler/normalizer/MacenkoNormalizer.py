"""
MacenkoNormalizer.py
Implement MacenkoNormalizer class for applying macenko normalization to H&E images.


Original macenko normalization code from:
https://github.com/schaugf/HEnorm_python
Modified by Isaiah Pressman (@IsaiahPressman)

Class Implementation: Jean R Clemenceau
Date Created: 01/11/2021
"""

import numpy as np
from wsitiler.normalizer.Normalizer import Normalizer

class MacenkoNormalizer(Normalizer):
    """
    Stain normalization object implementing Macenko stain normalization.
    This class is inherited from wsitiler.Normalizer
    """

    def __init__(self):
        #Inherit attributes from parent
        super().__init__(method_name="macenko")

        # Default values
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    @staticmethod
    def get_HE_maxC(img, Io=240, alpha=1, beta=0.15):
        # reshape image
        img = img.reshape((-1, 3))
        # calculate optical density
        OD = -np.log((img.astype(np.float) + 1) / Io)
        # remove transparent pixels
        ODhat = OD[~np.any(OD < beta, axis=1)]
        # compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:, 1:3])
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T

        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T
        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

        return HE, maxC

    @staticmethod
    def standardize_brightness(img):
        p = np.percentile(img, 90)
        return np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)

    def fit(self, reference_img, **kwargs):
        # reference_img = MacenkoNormalizer.standardize_brightness(reference_img)
        self.HERef, self.maxCRef = MacenkoNormalizer.get_HE_maxC(reference_img, **kwargs)
        self.is_fit = True

    def transform(self, img, get_H_E_results=False, Io=240, alpha=1, beta=0.15):
        """
        Normalize staining appearence of H&E stained images
        Input:
            I: RGB input image
            Io: (optional) transmitted light intensity

        Output:
            Inorm: normalized image
            Optional: (get_H_E_results)
                H: hematoxylin image
                E: eosin image

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        """
        # img = MacenkoNormalizer.standardize_brightness(img)
        # define height and width of image
        h, w, c = img.shape
        # reshape image
        img = np.array(img)
        img = img.reshape((-1, 3))
        # calculate optical density
        OD = -np.log((img.astype(np.float) + 1) / Io)
        # remove transparent pixels
        ODhat = OD[~np.any(OD < beta, axis=1)]
        # compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:, 1:3])
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T

        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T
        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
        tmp = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

        # unmix hematoxylin and eosin
        H = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
        H[H > 255] = 254
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

        E = np.multiply(Io, np.exp(np.expand_dims(-self.HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
        E[E > 255] = 254
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

        if get_H_E_results == True:
            return Inorm, H, E
        else:
            return Inorm

    def __repr__(self):
            return super().__repr__()