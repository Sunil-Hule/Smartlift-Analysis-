import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

# Drop the columns "participant", "category", and "set" from the dataframe 'df'
# and store the result in 'df_train'.
df_train = df.drop(["participant", "category", "set"], axis=1)

# Separate the features and the target variable.
# 'X' contains all columns except "label".
# 'y' contains only the "label" column.
X = df_train.drop("label", axis=1)
y = df_train["label"]

# Split the dataset into training and testing sets.
# 75% of the data is used for training and 25% is used for testing.
# 'stratify=y' ensures that the training and testing sets have similar
# distributions of the 'label' column.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Plotting the distribution of the "label" column.
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the total count of each unique value in the "label" column.
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)

# Plot the count of each unique value in the "label" column for the training set.
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")

# Plot the count of each unique value in the "label" column for the testing set.
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")

# Add a legend to differentiate the three bars for each unique value.
plt.legend()
plt.show()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
# Define basic features which probably represent accelerometer and gyroscope data.
basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

# Define squared features, likely representing magnitudes of accelerometer and gyroscope.
square_features = ["acc_r", "gyr_r"]

# Define features resulting from Principal Component Analysis (PCA).
pca_features = ["pca_1", "pca_2", "pca_3"]

# Extract time-related features from the dataset by looking for columns with '_temp_' in their name.
time_features = [f for f in df_train.columns if "_temp_" in f]

# Extract frequency-related features and the phase spectral entropy from the dataset.
freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]

# Define a cluster feature.
cluster_features = ["cluster"]

# Print the count of each feature set.
print("Basic Features:", len(basic_features))
print("Square Features:", len(square_features))
print("PCA Features:", len(pca_features))
print("Time Features:", len(time_features))
print("Freq Features:", len(freq_features))
print("Cluster Features:", len(cluster_features))

# Create different combinations of feature sets.
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
learner = ClassificationAlgorithms()
max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

selected_features = [
    "pca_1",
    "duration",
    "acc_z_freq_0.0_Hz_ws_14",
    "acc_y_temp_mean_ws_5",
    "gyr_x_freq_1.071_Hz_ws_14",
    "acc_x_freq_weighted",
    "acc_z_freq_weighted",
    "gyr_y",
    "acc_r",
    "gyr_z_freq_2.5_Hz_ws_14",
]
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------
possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected Features",
]

# Setting the number of iterations for training non-deterministic classifiers
iterations = 1

# Initialize a DataFrame to store the model accuracy results for each feature set
score_df = pd.DataFrame()

# Iterate over each feature set defined in 'possible_feature_sets'
for i, f in zip(range(len(possible_feature_sets)), feature_names):
    # Output the current feature set number being processed
    print("Feature set:", i)

    # Select the features for the current set for both training and testing datasets
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # Initialize performance counters for non-deterministic classifiers
    performance_test_nn = 0
    performance_test_rf = 0

    # For non-deterministic classifiers, train multiple times to average their scores
    for it in range(0, iterations):
        # Train the Neural Network model
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        # Accumulate the accuracy for the Neural Network
        performance_test_nn += accuracy_score(y_test, class_test_y)

        # Train the Random Forest model
        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        # Accumulate the accuracy for the Random Forest
        performance_test_rf += accuracy_score(y_test, class_test_y)

    # Compute the average performance for non-deterministic classifiers
    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # Train the deterministic classifiers:

    # Train the Decision Tree model
    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    # Train the Naive Bayes model
    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)
    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe

    # Define the list of models used in this iteration
    models = ["NN", "RF", "DT", "NB"]  # Removed KNN

    # Store the accuracy results for each model in the new DataFrame
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )

    # Append the new scores to the overall results DataFrame
    score_df = pd.concat([score_df, new_scores])


# possible_feature_sets = [
#     feature_set_1,
#     feature_set_2,
#     feature_set_3,
#     feature_set_4,
#     selected_features,
# ]

# feature_names = [
#     "Feature Set 1",
#     "Feature Set 2",
#     "Feature Set 3",
#     "Feature Set 4",
#     "Selected Features",
# ]

# iterations = 1
# score_df = pd.DataFrame()

# for i, f in zip(range(len(possible_feature_sets)), feature_names):
#     print("Feature set:", i)
#     selected_train_X = X_train[possible_feature_sets[i]]
#     selected_test_X = X_test[possible_feature_sets[i]]

#     # First run non deterministic classifiers to average their score.
#     performance_test_nn = 0
#     performance_test_rf = 0

#     for it in range(0, iterations):
#         print("\tTraining neural network,", it)
#         (
#             class_train_y,
#             class_test_y,
#             class_train_prob_y,
#             class_test_prob_y,
#         ) = learner.feedforward_neural_network(
#             selected_train_X,
#             y_train,
#             selected_test_X,
#             gridsearch=False,
#         )
#         performance_test_nn += accuracy_score(y_test, class_test_y)

#         print("\tTraining random forest,", it)
#         (
#             class_train_y,
#             class_test_y,
#             class_train_prob_y,
#             class_test_prob_y,
#         ) = learner.random_forest(
#             selected_train_X, y_train, selected_test_X, gridsearch=True
#         )
#         performance_test_rf += accuracy_score(y_test, class_test_y)

#     performance_test_nn = performance_test_nn / iterations
#     performance_test_rf = performance_test_rf / iterations

#     # And we run our deterministic classifiers:
#     print("\tTraining KNN")
#     (
#         class_train_y,
#         class_test_y,
#         class_train_prob_y,
#         class_test_prob_y,
#     ) = learner.k_nearest_neighbor(
#         selected_train_X, y_train, selected_test_X, gridsearch=True
#     )
#     performance_test_knn = accuracy_score(y_test, class_test_y)

#     print("\tTraining decision tree")
#     (
#         class_train_y,
#         class_test_y,
#         class_train_prob_y,
#         class_test_prob_y,
#     ) = learner.decision_tree(
#         selected_train_X, y_train, selected_test_X, gridsearch=True
#     )
#     performance_test_dt = accuracy_score(y_test, class_test_y)

#     print("\tTraining naive bayes")
#     (
#         class_train_y,
#         class_test_y,
#         class_train_prob_y,
#         class_test_prob_y,
#     ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

#     performance_test_nb = accuracy_score(y_test, class_test_y)

#     # Save results to dataframe
#     models = ["NN", "RF", "KNN", "DT", "NB"]
#     new_scores = pd.DataFrame(
#         {
#             "model": models,
#             "feature_set": f,
#             "accuracy": [
#                 performance_test_nn,
#                 performance_test_rf,
#                 performance_test_knn,
#                 performance_test_dt,
#                 performance_test_nb,
#             ],
#         }
#     )
#     score_df = pd.concat([score_df, new_scores])
# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
# Sort the score DataFrame based on accuracy in descending order
score_df.sort_values(by="accuracy", ascending=False)

# Plotting model accuracy for different feature sets
plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
# Typo corrected here from ylable to ylabel
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

# ---------------------------------------
# Select best model and evaluate results
# ---------------------------------------

# Train and evaluate the best model (Random Forest) on feature_set_4
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)
accuracy = accuracy_score(y_test, class_test_y)

# Create a confusion matrix for the best model
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# Plotting the confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Add text labels to the confusion matrix plot
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# ------------------------------------------------
# Select train and test data based on participant
# ------------------------------------------------

# Filtering out data based on the participant "A"
participant_df = df.drop(["set", "category"], axis=1)
X_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]
X_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
y_test = participant_df[participant_df["participant"] == "A"]["label"]

# Removing the 'participant' column after splitting
X_train = X_train.drop(["participant"], axis=1)
X_test = X_test.drop(["participant"], axis=1)

# Plotting the distribution of labels in the training and testing sets
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Use best model again and evaluate results on the new dataset
# --------------------------------------------------------------

# Training and evaluating the Random Forest model again on the new dataset
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)
accuracy = accuracy_score(y_test, class_test_y)

# Create a confusion matrix for the evaluated model
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# Plotting the confusion matrix again for the new dataset
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Add text labels to the confusion matrix plot
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=False
)
accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
