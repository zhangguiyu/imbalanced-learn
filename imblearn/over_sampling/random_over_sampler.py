"""Class to perform random over-sampling."""
from __future__ import division, print_function

from collections import Counter

import numpy as np
from sklearn.utils import check_random_state

from ..base import BaseMulticlassSampler


class RandomOverSampler(BaseMulticlassSampler):
    """Class to perform random over-sampling.

    Object to over-sample the minority class(es) by picking samples at random
    with replacement.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If ratio is float, K new samples are generated:
            K = ratio * C_i - C_j
        where
            C_i = number of samples in majority class i
            C_j = number of samples in all other class j where j != i

        If specifying ratio, ensure that 

            C_k / C_i <= ratio <= 1.0
        
        where C_k is number of samples of the second largest class.

        Setting ratio='auto' is equivalent to setting ratio = 1, i.e.,
            K = C_i - C_j 

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    Supports multiple classes.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
    RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> ros = RandomOverSampler(random_state=42)
    >>> X_res, y_res = ros.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 900, 1: 900})

    """

    def __init__(self, ratio='auto', random_state=None):

        super(RandomOverSampler, self).__init__(
            ratio=ratio, random_state=random_state)

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """

        # Set to True to enable downsampling for classes that exceed 
        # ratio * | C_max |
        # setting to False will just leave those classes as is
        downsample = False

        # Define the number of sample to create
        if self.ratio == 'auto':
            ratio = 1.0
        else:
            ratio = self.ratio

        # Keep the samples from the majority class
        if ratio==1: 
            X_resampled = X[y == self.maj_c_]
            y_resampled = y[y == self.maj_c_]
        else:       # optionally downsampled
            X_resampled = None
            y_resampled = None

        # Loop over the other classes over picking at random
        for key in self.stats_c_.keys():

            # If this is the majority class, skip it
            if key == self.maj_c_:
                if ratio == 1:
                    continue

            target_size = self.ratio * self.stats_c_[self.maj_c_]
            num_samples = int( target_size - self.stats_c_[key] )

            if (num_samples < 0):
                if downsample:
                    pick = int(target_size)
                else:
                    pick = 0
            else:
                pick = int(num_samples)

            # Pick some elements at random
            high=self.stats_c_[key]
            random_state = check_random_state(self.random_state)
            indx = random_state.randint(low=0, high=high, size=pick)

            if X_resampled is None:
                Xbase = []
                ybase = []
            else:
                Xbase = [ X_resampled ]
                ybase = [ y_resampled ]

            if num_samples < 0: # existing class size > ratio * |C_i|
                if downsample:
                    Xbase.append(X[y==key][indx])
                    ybase.append(y[y==key][indx])
                else:   # class size unchanged
                    Xbase.append(X[y==key])
                    ybase.append(y[y==key])
            elif num_samples == 0:   # majority or equal sized class
                Xbase.append(X[y==key]) 
                ybase.append(y[y==key]) 
            else:   # oversample
                Xbase.append(X[y==key])
                Xbase.append(X[y==key][indx])
                ybase.append(y[y==key])
                ybase.append(y[y==key][indx])

            X_resampled = np.concatenate(Xbase, axis=0)
            y_resampled = np.concatenate(ybase, axis=0)

        self.logger.info('Over-sampling performed: %s', Counter(y_resampled))

        return X_resampled, y_resampled
