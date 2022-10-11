"""
Normalizer.py
Implement the parent class for all WSItiler Normalizers

Author: Jean R Clemenceau
Date Created: 10/11/2022
"""

class Normalizer:
    """
    Parent class for all WSItiler normalizer methods.

    Every child must set a method name, model fit status, and
    implement the fit() and transform() functions.
    """

    def __init__(self, method_name: str="no_norm"):
        """
        Normalizer class constructor. 
        
        Must set 'method' attribute and initialize is_fit attribute to False.
        """
        # Name of the Normalizing method
        self.method = method_name
        self.is_fit = False
        return

    def fit(self):
        """
        Fits Normalizer's model to a reference image.
        Any child class is required to implement this function.

        Input:
            reference_image

        Output:
            None

        Note: Once fit, function must set is_fit attribute to True.
        """
        self.is_fit = True
        raise NotImplementedError("ERROR: is_fit() was not implemented in Normalizer:%s" % self.method)

    def transform(self):
        """
        Applies Normalizer method to source image.
        Any child class is required to implement this function.

        Input:
            img: Source image

        Output:
            Transformed image.
        """
        raise NotImplementedError("ERROR: transform() was not implemented in Normalizer:%s" % self.method)

    def __repr__(self):
        return f'wsitiler.Normalizer({self.method}: {"Fit" if self.is_fit else "Not Fit"})'