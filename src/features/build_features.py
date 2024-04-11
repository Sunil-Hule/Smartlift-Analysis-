import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
# Load the data after outliers have been removed using Chauvenet's criterion
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

# Extract the first 6 columns of the dataframe to be used as predictor columns
predictor_columns = list(df.columns[:6])

# Setting the plot style for better aesthetics
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)  # Define the figure size
plt.rcParams["figure.dpi"] = 100  # Define the dots-per-inch (dpi) value for the plot
plt.rcParams["lines.linewidth"] = 2  # Set the line width for plots

# Display basic information about the dataframe, including column types and non-null values
df.info()

# Plot 'gyr_y' values for the subset where 'set' equals 35
subset = df[df["set"] == 35]["gyr_y"].plot()

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

# Iterate through each predictor column and interpolate missing values
for col in predictor_columns:
    df[col] = df[col].interpolate()

# Display the dataframe information again to verify changes post imputation
df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

# Plot 'acc_y' values for subsets where 'set' equals 25 and 50 respectively
df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

# Calculate the duration for set 1
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]

# Display the duration in seconds for set 1
duration.seconds

# Calculate the duration for each unique set and store it in a new column 'duration'
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

# Calculate the mean duration for each 'category' and store it in 'duration_df'
duration_df = df.groupby(["category"])["duration"].mean()

# Display the mean duration for the first category divided by 5
duration_df.iloc[0] / 5

# Display the mean duration for the second category divided by 10
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

# Copy the original dataframe for applying lowpass filter operations
df_lowpass = df.copy()

# Initialize the LowPassFilter
LowPass = LowPassFilter()

# Define sampling frequency and cutoff frequency for the filter
fs = 1000 / 200
# Derived from a division operation (adjust comment if derivation logic changes)
cutoff = 1.3

# Apply a low-pass filter on the 'acc_y' column of the data
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

# Extract a subset of the data where 'set' equals 45
subset = df_lowpass[df_lowpass["set"] == 45]

# Print the label of the first entry in the subset
print(subset["label"][0])

# Create a plot comparing the original 'acc_y' data with its low-pass filtered counterpart
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# Apply the low-pass filter on all predictor columns and update the original columns with filtered values
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

# Create a copy of the low-pass filtered data for PCA operations
df_pca = df_lowpass.copy()

# Initialize the Principal Component Analysis (PCA) class
PCA = PrincipalComponentAnalysis()

# Determine the explained variance for each principal component
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# Plot the explained variance for each principal component
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Principal component number")
plt.ylabel("explained Variance")
plt.show()

# Apply PCA to the data and keep only the first 3 principal components
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# Extract a subset of the PCA-applied data where 'set' equals 35
subset = df_pca[df_pca["set"] == 35]

# Plot the first three principal components for the subset
subset[["pca_1", "pca_2", "pca_3"]].plot()

# Display the PCA-applied dataframe
df_pca


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
# Copy the PCA-processed dataframe for squared operations
df_squared = df_pca.copy()

# Calculate the squared magnitudes for accelerometer and gyroscope readings
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

# Calculate the root of the squared magnitudes to get the resultant magnitudes and add them to the dataframe
df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# Extract a subset of the squared dataframe where 'set' equals 14
subset = df_squared[df_squared["set"] == 14]

# Plot the resultant magnitudes for the accelerometer and gyroscope for the subset
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# Display the dataframe with resultant magnitudes
df_squared

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

# Copy the squared dataframe for temporal abstraction operations
df_temporal = df_squared.copy()

# Initialize the NumericalAbstraction
NumAbs = NumericalAbstraction()

# Add the resultant magnitudes to the list of predictor columns
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

# Define the window size for the abstraction operation
ws = int(1000 / 200)

# Abstract the data using both mean and standard deviation over the defined window size
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

# Display the temporally abstracted dataframe
df_temporal

# Create a list to store subsets of the abstracted data for each unique 'set' value
df_temporal_list = []

# Temporally abstract the data for each unique 'set' value
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

# Concatenate all the subsets into a single dataframe
df_temporal = pd.concat(df_temporal_list)

# Plot the original, mean, and standard deviation values of accelerometer's y-axis for the subset
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()

# Plot the original, mean, and standard deviation values of gyroscope's y-axis for the subset
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
# Create a copy of the temporal dataframe and reset its index
df_freq = df_temporal.copy().reset_index()

# Initialize the FourierTransformation
FreqAbs = FourierTransformation()

# Define the sampling frequency and window size for frequency abstraction
fs = int(1000 / 200)
ws = int(2800 / 200)

# Apply Fourier transformation to abstract the 'acc_y' frequency domain characteristics
df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

# Display the column names of the frequency abstracted dataframe
df_freq.columns

# Extract a subset of the frequency abstracted dataframe for visualization
subset = df_freq[df_freq["set"] == 15]
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

# List to store frequency abstracted subsets
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

# Concatenate all subsets into a single dataframe and set the index to "epoch (ms)"
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# Remove any rows containing NA values
df_freq = df_freq.dropna()

# Filter out overlapping windows by skipping every other row
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

# Copy the frequency dataframe for clustering operations
df_cluster = df_freq.copy()
cluster_columns = ["acc_y", "acc_y", "acc_z"]

# Define a range of k-values to determine the optimal number of clusters
k_values = range(2, 10)
inertias = []

# Perform k-means clustering for each k-value and record the sum of squared distances (inertia)
for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

# Plot the sum of squared distances for each k-value to determine the optimal number of clusters
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

# 3D scatter plot visualization of the clusters based on accelerometer readings
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
colors = ["r", "g", "b", "y", "c"]

for idx, c in enumerate(df_cluster["cluster"].unique()):
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        color=colors[idx],
        label=f"Cluster {c}",
    )

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# 3D scatter plot visualization based on labels
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for idx, c in enumerate(df_cluster["label"].unique()):
    subset = df_cluster[df_cluster["label"] == c]
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=f"label {c}",
    )

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
