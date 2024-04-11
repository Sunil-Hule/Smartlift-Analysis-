##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

# Updated by Efthimios Vlahos on 9-12-2023

from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy
import pandas as pd


# This class removes the high frequency data (that might be considered noise) from the data.
# We can only apply this when we do not have missing values (i.e. NaN).
# This class contains methods to apply a low-pass filter to the data.
class LowPassFilter:
    # This function applies a low-pass filter to the given data.
    def low_pass_filter(
        self,
        data_table,
        col,
        sampling_frequency,
        cutoff_frequency,
        order=5,
        phase_shift=True,
    ):
        # The Nyquist frequency is half the sampling frequency.
        # It's the maximum frequency that can be represented without aliasing.
        nyq = 0.5 * sampling_frequency
        # Normalize the cutoff frequency by dividing by the Nyquist frequency.
        cut = cutoff_frequency / nyq

        # Use the Butterworth filter to get the filter coefficients.
        b, a = butter(order, cut, btype="low", output="ba", analog=False)

        # If phase_shift is True, apply the filter without phase shift using filtfilt.
        # Else, use lfilter which might introduce a phase shift.
        if phase_shift:
            data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])

        # Return the filtered data.
        return data_table


# This class contains methods for applying Principal Component Analysis (PCA) to the data.
class PrincipalComponentAnalysis:
    # Constructor: Initialize the pca attribute as an empty list.
    def __init__(self):
        self.pca = []

    # Normalize the dataset: For each column in 'columns',
    # subtract the mean of the column and divide by the range (max-min).
    def normalize_dataset(self, data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max() - data_table[col].min()
            )
        return dt_norm

    # Determine the explained variance of each principal component.
    def determine_pc_explained_variance(self, data_table, cols):
        # First, normalize the data.
        dt_norm = self.normalize_dataset(data_table, cols)

        # Then, perform the PCA using the sklearn PCA class.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])

        # Return the explained variance ratios of the components.
        return self.pca.explained_variance_ratio_

    # Apply PCA on the data and add the principal components as new columns.
    def apply_pca(self, data_table, cols, number_comp):
        # Normalize the data.
        dt_norm = self.normalize_dataset(data_table, cols)

        # Perform PCA with the desired number of components.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform the original data to get the principal components.
        new_values = self.pca.transform(dt_norm[cols])

        # Add each principal component as a new column to the data.
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        # Return the dataset with the new PCA columns.
        return data_table
