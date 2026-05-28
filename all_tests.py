import os

from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd
import math

from bird_object_main import Bird

"""
Filename: all_tests.py
Date: 22-01-2026
Version: 1.0.1
Description: Run all the tests that generated the figures found in the paper
"""


def create_test_excel_sheets(excels_path, test_clips_excel_path, other_folder_path):
    """Create excel spreadsheets for all the possible tests"""

    # Use the 5 second clips master sheet to create the test excel sheets
    df = pd.read_excel(test_clips_excel_path)
    filenames = list(df["Filename"])
    videos = list(df["Video"])
    bird_actual = list(df["Bird actual"])
    motion_actual = list(df["Motion actual"])

    roc_test = {'Filename': filenames, 'Video': videos, 'actual': bird_actual}
    pd.DataFrame(roc_test).to_excel(excels_path + "5 second clips m1 all thresholds test.xlsx", index=False,
                                    header=True)
    pd.DataFrame(roc_test).to_excel(excels_path + "5 second clips up to m2 all thresholds test.xlsx",
                                    index=False,
                                    header=True)

    m2_bird_test = {'Filename': filenames, 'actual': bird_actual}
    pd.DataFrame(m2_bird_test).to_excel(excels_path + "5 second clips m2 single threshold bird test.xlsx", index=False,
                                        header=True)
    pd.DataFrame(m2_bird_test).to_excel(excels_path + "5 second clips m2 all thresholds bird test.xlsx", index=False)

    m2_action_test = {'Filename': filenames, 'actual': motion_actual}
    pd.DataFrame(m2_action_test).to_excel(excels_path + "5 second clips m2 single threshold action test.xlsx",
                                          index=False, header=True)

    # Use the other flying objects master sheet to create the test excel sheets
    other_filenames = []
    other_actual = []
    for filename in os.listdir(other_folder_path):
        other_filenames.append(other_folder_path + filename)
        other_actual.append("non")

    other_test = {'Filename': other_filenames, 'actual': other_actual}
    pd.DataFrame(other_test).to_excel(excels_path + "Other flying objects m1 all thresholds test.xlsx", index=False,
                                      header=True)
    pd.DataFrame(other_test).to_excel(excels_path + "Other flying objects m2 all thresholds test.xlsx", index=False,
                                      header=True)


def generate_roc(excel_path, thresholds):
    """Generate true positive rate and false positive rate points for an ROC"""

    df = pd.read_excel(excel_path)
    actual = df["actual"]

    roc_tpr = []
    roc_fpr = []

    for threshold in thresholds:
        predicted = df[str(threshold)]

        confusion_matrix = metrics.confusion_matrix(list(actual), list(predicted), labels=["bird", "non"])

        tp = confusion_matrix[0][0]
        fn = confusion_matrix[0][1]
        fp = confusion_matrix[1][0]
        tn = confusion_matrix[1][1]

        print(tp, fp, fn, tn)

        total_bird = tp + fn
        total_non = fp + tn

        tpr = tp / total_bird
        fpr = fp / total_non

        roc_tpr.append(tpr)
        roc_fpr.append(fpr)

    return roc_tpr, roc_fpr


def get_ideal_threshold_index(roc_tpr, roc_fpr):
    """Get the list index of the ideal threshold for an ROC"""

    # calculate all the distances between points on the ROC curve and the coordinate (0,1)
    distances = []
    for i in range(len(roc_tpr)):
        coordinate = (roc_fpr[i], roc_tpr[i])
        dist = math.dist(coordinate, (0, 1))
        print(dist, thresholds[i])
        distances.append(dist)

    # Determine the minimum distance and its index in the list of distances
    min_distance = distances[0]
    min_distance_index = 0
    for i in range(len(distances)):
        if distances[i] <= min_distance:
            min_distance = distances[i]
            min_distance_index = i

    return min_distance_index


def calculate_auc(roc_excel):
    """Calculate the area under the ROC curve"""

    # Get the points for the ROC curve
    roc_tpr, roc_fpr = generate_roc(roc_excel, thresholds)

    # Sort the coordinates so that they are in order of smallest to largest false positive rate
    roc_fpr, roc_tpr = zip(*sorted(zip(roc_fpr, roc_tpr)))

    # Calculate area under the curve using trapeziums
    auc = 0
    for i in range(len(roc_fpr) - 1):
        trapezium = 0.5 * (roc_tpr[i] + roc_tpr[i + 1]) * (roc_fpr[i + 1] - roc_fpr[i])
        auc += trapezium
    return auc


def test_single_vid_species(vid_path):
    """Return the species and associated confidence score for one input video"""

    bird_object = Bird(vid_path)
    species, score = bird_object.get_species(False)

    return species, score


def model1_roc_test(folder_path, excel_path, thresholds):
    """Get Model 1 predictions for all thresholds and write to an excel sheet"""

    # Get Model 1 predictions for all thresholds
    bird_object = Bird(folder_path)
    actual, all_predicted = bird_object.test_model1(excel_path, thresholds)

    print(all_predicted)

    # Write predictions for all thresholds to an excel sheet
    df = pd.read_excel(excel_path)
    for p in range(len(thresholds)):
        predicted = all_predicted[:, p]

        df[str(thresholds[p])] = predicted
        df.to_excel(excel_path, header=True, index=False)


def only_m2_roc_test(folder_path, excel_path, thresholds):
    """Get Model 2 predictions for all thresholds and write to an excel sheet"""

    # Get Model 2 predictions for all thresholds
    bird_object = Bird(folder_path)
    actual, predicted_per_threshold = bird_object.test_model2(excel_path, thresholds, False)

    # Write predictions for all thresholds to an excel sheet
    df = pd.read_excel(excel_path)
    for p in range(len(predicted_per_threshold)):
        predicted = predicted_per_threshold[p]

        df[str(thresholds[p])] = predicted
        df.to_excel(excel_path, header=True, index=False)


def test_up_to_m2(folder_path, m1_excel_path, up_to_m2_excel_path, thresholds, m2_threshold):
    m1_roc_tpr, m1_roc_fpr = generate_roc(m1_excel_path, thresholds)
    m1_ideal_threshold = thresholds[get_ideal_threshold_index(m1_roc_tpr, m1_roc_fpr)]

    bird_object = Bird(folder_path)
    predicted, m2_threshold = bird_object.test_up_to_m2(m1_excel_path, m1_ideal_threshold, m2_threshold)

    df = pd.read_excel(up_to_m2_excel_path)
    df[str(m2_threshold)] = predicted
    df.to_excel(up_to_m2_excel_path, header=True, index=False)


def combine_excel_sheets(excel_path, other_excel_path, thresholds):
    """Combine the predictions for other flying objects with the predictions for the 5-second clips test frames into
    one Excel sheet"""

    # Read the Excel sheet for the 5-second clips test frames dataset
    df = pd.read_excel(excel_path)

    # Read the Excel sheet for the other flying objects dataset
    df_other = pd.read_excel(other_excel_path)

    # Get the actual classes for the 5-second clips test frames
    actual = list(df["actual"])

    # Get the actual classes for the other flying objects
    actual_other = list(df_other["actual"])

    # Append the other flying objects classes to the 5-second clips test frames classes
    actual.extend(actual_other)

    # Read the filenames of the 5-second clips test frames dataset
    filenames = list(df["Filename"])

    # Read the filenames of the other flying objects dataset
    filenames_other = list(df_other["Filename"])

    # Append the filenames of the other flying objects to the filenames of the 5-second clips test frames
    filenames.extend(filenames_other)

    # Create a combined DataFrame with the combined filenames and actual classes lists
    df_combined = pd.DataFrame({"Filename": filenames, "actual": actual})

    # Cycle through all thresholds
    for threshold in thresholds:

        # Get the predictions for a single threshold from the 5-second clips test frames Excel sheet
        predicted = list(df[str(threshold)])

        # Get the predictions for a single threshold from the other flying objects Excel sheet
        predicted_other = list(df_other[str(threshold)])

        # Append the predictions for the other flying objects to the predictions for the 5-second clips test frames
        predicted.extend(predicted_other)

        # Write the combined predictions list to the combined Dataframe
        df_combined[str(threshold)] = predicted

    # Write the combined DataFrame to the other flying objects Excel sheet
    df_combined.to_excel(other_excel_path, header=True, index=False)


def m2_test_single_threshold_action(folder_path, excel_path, threshold):
    """Get Model 2 predictions for a single threshold and write an excel sheet"""

    df = pd.read_excel(excel_path)

    # Get Model 2 predictions for a single threshold
    bird_object = Bird(folder_path)
    actual, predicted_per_threshold = bird_object.test_model2(excel_path, [threshold], True)
    predicted = predicted_per_threshold[0]

    # Write predictions to an Excel sheet
    df["predicted"] = predicted
    print(predicted)
    df.to_excel(excel_path, header=True, index=False)


def plot_confusion_matrix(actual, predicted, title, predict_labels, display_labels, filename):
    """Generate and save confusion matrix from actual and predicted classes and save to file"""

    # Set parameters for the plot
    plt.rc('font', size=300)
    plt.rc('axes', titlesize=300)

    # Create plot
    fig, ax = plt.subplots(figsize=(120, 60))

    # Add title and axes labels with padding
    ax.set_xlabel('Predicted label', labelpad=200.0)
    ax.set_ylabel('True label', labelpad=200.0)
    ax.set_title(title, pad=150.0)

    # Set font size
    plt.rcParams.update({'font.size': 300})

    # Generate the confusion matrix from the lists of actual and predicted classes and set the order of labels
    confusion_matrix = metrics.confusion_matrix(list(actual), list(predicted), labels=predict_labels)

    # Create the plot of the confusion matrix and set the display labels
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)

    # Plot the confusion matrix
    cm_display.plot(ax=ax)

    # Save the plot to file
    plt.savefig(results_path + filename, bbox_inches="tight")

    # Calculate the metrics of the confusion matrix
    accuracy = metrics.accuracy_score(actual, predicted)
    precision = metrics.precision_score(actual, predicted, average="weighted")
    recall = metrics.recall_score(actual, predicted, average="weighted")

    print(accuracy, precision, recall)


def plot_combined_roc(all_excels_path, m1_excel_path, m2_excel_path, thresholds, results_path):
    """Generate combined ROC for Model 1, Model 1 + Object Tracking, Model 1 + Object Tracking + Model 2 from the Excel
    sheets for the 5-second clips dataset and save to file"""

    # Set the file paths of the ROC Excel sheets
    m1_roc_excel = all_excels_path + "m1 roc.xlsx"
    m2_roc_excel = all_excels_path + "m2 roc.xlsx"

    # Set the parameters of the ROC plot
    plt.rc('font', size=60)
    plt.rc('axes', titlesize=60)

    # Create plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # Set font size
    plt.rcParams.update({'font.size': 60})

    # Add title and axes labels with padding
    plt.xlabel('False positive rate', labelpad=40.0)
    plt.ylabel('True positive rate', labelpad=40.0)
    plt.title('Birds vs Empty', pad=50.0)

    # Generate M1 ROC from Excel sheet
    m1_roc_tpr, m1_roc_fpr = generate_roc(m1_excel_path, thresholds)

    # Write the ROC points to an Excel sheet
    df = pd.DataFrame({"Threshold": thresholds, "fpr": m1_roc_fpr, "tpr": m1_roc_tpr})
    df.to_excel(m1_roc_excel, header=True, index=False)

    # Calculate the ideal threshold of the M1 ROC
    ideal_threshold_index_m1 = get_ideal_threshold_index(m1_roc_tpr, m1_roc_fpr)
    print("IDEAL THRESHOLD OF M1 IS ", thresholds[ideal_threshold_index_m1])

    # Plot the M1 ROC line
    plt.plot(m1_roc_fpr, m1_roc_tpr, c="blue", label="Only M1", marker=MarkerStyle('o', fillstyle='full'),
             linewidth=10, markersize=25, zorder=1)
    # Highlight the M1 ideal threshold point in another colour
    plt.plot(m1_roc_fpr[ideal_threshold_index_m1], m1_roc_tpr[ideal_threshold_index_m1], c="red",
             marker=MarkerStyle('o', fillstyle='full'),
             markersize=35, zorder=2)

    # Generate M2 ROC from Excel sheet
    m2_roc_tpr, m2_roc_fpr = generate_roc(m2_excel_path, thresholds)

    # Write the ROC points to an Excel sheet
    df = pd.DataFrame({"Threshold": thresholds, "fpr": m2_roc_fpr, "tpr": m2_roc_tpr})
    df.to_excel(m2_roc_excel, header=True, index=False)

    # Calculate the ideal threshold of the M2 ROC
    ideal_threshold_index_m2 = get_ideal_threshold_index(m2_roc_tpr, m2_roc_fpr)
    print("IDEAL THRESHOLD OF M2 IS ", thresholds[ideal_threshold_index_m2])

    # Plot the M2 ROC line
    plt.plot(m2_roc_fpr, m2_roc_tpr, c="orange", marker=MarkerStyle('o', fillstyle='full'),
             label="Only M2", linewidth=10, markersize=25, zorder=1)
    # Highlight the M2 ideal threshold point in another colour
    plt.plot(m2_roc_fpr[ideal_threshold_index_m2], m2_roc_tpr[ideal_threshold_index_m2], c="red",
             marker=MarkerStyle('o', fillstyle='full'),
             markersize=35, zorder=2)

    # Create a legend
    plt.legend(loc="lower right")

    # Save the plot to file
    plt.savefig(results_path + "combined roc.png", bbox_inches="tight")

    # plt.show()


def plot_combined_roc_other(all_excels_path, m1_other_excel_path, m2_other_excel_path, thresholds, results_path):
    """Generate combined ROC for M1 and M2 from the excel sheets for the 5 second clips + other flying objects dataset
    and save to file"""

    # Set the file paths of the ROC Excel sheets
    m1_other_roc_excel = all_excels_path + "m1 other roc.xlsx"
    m2_other_roc_excel = all_excels_path + "m2 other roc.xlsx"

    # Set the parameters of the ROC plot
    plt.rc('font', size=60)
    plt.rc('axes', titlesize=60)

    # Create plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # Set font size
    plt.rcParams.update({'font.size': 60})

    # Add title and axes labels with padding
    plt.xlabel('False positive rate', labelpad=40.0)
    plt.ylabel('True positive rate', labelpad=40.0)
    plt.title('Birds vs Other Flying Objects', pad=30.0)

    # Generate M1 ROC from Excel sheet
    m1_roc_tpr, m1_roc_fpr = generate_roc(m1_other_excel_path, thresholds)

    # Write the ROC points to an Excel sheet
    df = pd.DataFrame({"Threshold": thresholds, "fpr": m1_roc_fpr, "tpr": m1_roc_tpr})
    df.to_excel(m1_other_roc_excel, header=True, index=False)

    # Calculate the ideal threshold of the M1 ROC
    ideal_threshold_index = get_ideal_threshold_index(m1_roc_tpr, m1_roc_fpr)

    # Plot the M1 ROC line
    plt.plot(m1_roc_fpr, m1_roc_tpr, c="blue", label="Only M1", marker=MarkerStyle('o', fillstyle='full'),
             linewidth=10, markersize=25, zorder=1)

    # Highlight the M1 ideal threshold point in another colour
    plt.plot(m1_roc_fpr[ideal_threshold_index], m1_roc_tpr[ideal_threshold_index], c="red",
             marker=MarkerStyle('o', fillstyle='full'),
             markersize=35, zorder=2)

    # Generate M2 ROC from Excel sheet
    m2_roc_tpr, m2_roc_fpr = generate_roc(m2_other_excel_path, thresholds)

    # Write the ROC points to an Excel sheet
    df = pd.DataFrame({"Threshold": thresholds, "fpr": m2_roc_fpr, "tpr": m2_roc_tpr})
    df.to_excel(m2_other_roc_excel, header=True, index=False)

    # Plot the M2 ROC line
    plt.plot(m2_roc_fpr, m2_roc_tpr, c="orange", label="Only M2", marker=MarkerStyle('o', fillstyle='full'),
             linewidth=10, markersize=25, zorder=1)

    # Create a legend
    plt.legend(loc="lower right")

    # Save the plot to file
    plt.savefig(results_path + "combined roc other.png", bbox_inches="tight")

    # plt.show()


def plot_bar_chart(m1_excel_path, up_to_m2_excel_path, threshold, m2_ideal_threshold):
    """Plot bar chart of accuracy, precision and specificity of Model 1, Model 1 + Object Tracking, and
    Model 1 + Object Tracking + Model 2 for one threshold from excel sheets for 5 second clips dataset and save to
    file"""

    # Calculate accuracy, precision and specificity of Model 1 for one threshold from the Excel sheet

    m1_df = pd.read_excel(m1_excel_path)
    actual_m1 = m1_df["actual"]
    predicted_m1 = m1_df[str(threshold)]

    confusion_matrix_m1 = metrics.confusion_matrix(actual_m1, predicted_m1)

    precision_m1 = metrics.precision_score(actual_m1, predicted_m1, pos_label='bird', average='binary')
    recall_m1 = metrics.recall_score(actual_m1, predicted_m1, pos_label='bird', average='binary')
    specificity_m1 = confusion_matrix_m1[1][1] / (confusion_matrix_m1[1][1] + confusion_matrix_m1[1][0])

    print(precision_m1, recall_m1, specificity_m1)

    # Calculate accuracy, precision and specificity of Model 1 + Object Tracking + Model 2 for one threshold from the
    # Excel sheet

    up_to_m2_df = pd.read_excel(up_to_m2_excel_path)
    actual_m2 = up_to_m2_df["actual"]
    predicted_m2 = up_to_m2_df[str(threshold)]

    for i in range(len(list(actual_m2))):
        if actual_m2[i] == "upstroke" or actual_m2[i] == "downstroke":
            actual_m2[i] = "bird"
        if predicted_m2[i] == "upstroke" or predicted_m2[i] == "downstroke":
            predicted_m2[i] = "bird"

    confusion_matrix_m2 = metrics.confusion_matrix(actual_m2, predicted_m2)

    precision_m2 = metrics.precision_score(actual_m2, predicted_m2, pos_label='bird', average='binary')
    recall_m2 = metrics.recall_score(actual_m2, predicted_m2, pos_label='bird', average='binary')
    specificity_m2 = confusion_matrix_m2[1][1] / (confusion_matrix_m2[1][1] + confusion_matrix_m2[1][0])

    print(precision_m2, recall_m2, specificity_m2)

    # Generate bar chart and save to file

    # Set parameters of plot
    plt.rc('font', size=100)
    plt.rc('axes', titlesize=100)
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1.5

    # Create plot
    fig, ax = plt.subplots(figsize=(40, 22))

    stages = ["M1(" + str(threshold) + ")", "M1(" + str(threshold) + ")+M2(" + str(m2_ideal_threshold) + ")"]
    metrics_names = ["Precision", "Recall", "Specificity"]
    precision = [precision_m1, precision_m2]
    recall = [recall_m1, recall_m2]
    specificity = [specificity_m1, specificity_m2]
    print(precision, recall, specificity)

    bar_width = 0.25
    x = np.arange(len(stages))

    plt.bar(x - bar_width, precision, width=bar_width, label=metrics_names[0])
    plt.bar(x, recall, width=bar_width, label=metrics_names[1])
    plt.bar(x + bar_width, specificity, width=bar_width, label=metrics_names[2])
    plt.ylim(0.5, 1)

    plt.xlabel('Models used for bird detection', labelpad=40.0)
    plt.ylabel('Value', labelpad=40.0)
    plt.xticks(x, stages, rotation=10)
    leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.add_artist(leg)

    plt.tight_layout()

    plt.savefig(results_path + "bar chart.png", bbox_inches="tight")

    # plt.show()


def plot_flight_pattern_highlights(vid, species):
    """Generate flight pattern graph with flapping sections and switching points highlighted for one video"""

    bird_object = Bird(vid)
    bird_object.plot_pattern_highlights(species, results_path)


# Change "path/to" to the path to the git folder
folder_path = "path/to/automated-bird-flight-pattern-classification/"
datasets_path = folder_path + "Datasets/"
all_excels_path = datasets_path + "Dataset excel sheets/"
results_path = folder_path + "Results/"

thresholds = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
              0.95, 1.0]
m1_threshold = 0.1
m2_threshold = 0.

m1_excel_path = all_excels_path + "5 second clips m1 all thresholds test.xlsx"
m2_excel_path = all_excels_path + "5 second clips m2 all thresholds test.xlsx"
up_to_m2_excel_path = all_excels_path + "5 second clips up to m2 all thresholds test.xlsx"
m2_single_excel_path = all_excels_path + "5 second clips m2 single threshold test.xlsx"
m2_ff_excel_path = all_excels_path + "Full-frame m2 single threshold test.xlsx"
m1_other_excel_path = all_excels_path + "Other flying objects m1 all thresholds test.xlsx"
m2_other_excel_path = all_excels_path + "Other flying objects m2 all thresholds test.xlsx"

if not os.path.isdir(results_path):
    os.mkdir(results_path)

'''Uncomment below tests as required'''

# ONLY RUN ONCE BEFORE TESTING
# # # create_test_excel_sheets(all_excels_path, all_excels_path + "5 second clips test frames.xlsx",
# # #                          datasets_path + "Other flying objects/")
#
# SINGLE VID SPECIES TEST - CHANGE vid_path TO PATH TO INPUT VIDEO
# vid_path = datasets_path + "5 second vids m1 m2 test/SPECIES/VID_NAME.mp4'"
# species, score = test_single_vid_species(vid_path)
# print(species, score)
#
# MODEL 1 ROC TEST
# model1_roc_test(folder_path, m1_excel_path, thresholds)
#
# MODEL 2 ROC TEST
# only_m2_roc_test(folder_path, m2_excel_path, thresholds)
#
# UP TO M2 SINGLE THRESHOLD TEST
# test_up_to_m2(folder_path, m1_excel_path, up_to_m2_excel_path, thresholds, m2_threshold)
#
# MODEL 1 ROC TEST OTHER
# model1_roc_test(folder_path, m1_other_excel_path, thresholds)
#
# MODEL 2 ROC TEST OTHER
# only_m2_roc_test(folder_path, m2_other_excel_path, thresholds)
#
# MODEL 2 SINGLE THRESHOLD TEST
# m2_test_single_threshold_action(folder_path, m2_single_excel_path, m2_threshold)
#
# PLOT M2 SINGLE THRESHOLD CONFUSION MATRIX
# df = pd.read_excel(m2_single_excel_path)
# actual = df["actual"]
# predicted = df["predicted"]
#
# plot_confusion_matrix(actual, predicted, "M2: Birds vs Empty",
#                       ["downstroke", "non", "upstroke"], ["Down", "Non", "Up"],
#                       "m2 test matrix.png")
#
# # MODEL 2 FULL-FRAME TEST
# m2_test_single_threshold_action(folder_path, m2_ff_excel_path, m2_threshold)
#
# PLOT M2 FULL-FRAME CONFUSION MATRIX
# df = pd.read_excel(m2_ff_excel_path)
# actual = df["actual"]
# predicted = df["predicted"]
#
# plot_confusion_matrix(actual, predicted, "M2: Full Frames",
#                       ["downstroke", "upstroke"], ["Down", "Up"],
#                       "m2 ff test matrix.png")

# PLOT COMBINED ROC
# plot_combined_roc(all_excels_path, m1_excel_path, m2_excel_path, thresholds, results_path)
#
# CALCULATE AUC VALUES
# m1_auc = calculate_auc(m1_excel_path)
# print(m1_auc)
# m2_auc = calculate_auc(m2_excel_path)
# print(m2_auc)
#
# PLOT COMBINED ROC OTHER
# combine_excel_sheets(m1_excel_path, m1_other_excel_path, thresholds)
# combine_excel_sheets(m2_excel_path, m2_other_excel_path, thresholds)
# plot_combined_roc_other(all_excels_path, m1_other_excel_path, m2_other_excel_path, thresholds, results_path)
#
# CALCULATE AUC OTHER VALUES
# m1_other_auc = calculate_auc(m1_other_excel_path)
# print(m1_other_auc)
# m2_other_auc = calculate_auc(m2_other_excel_path)
# print(m2_other_auc)
#
# # BAR CHART
# m2_roc_tpr, m2_roc_fpr = generate_roc(m2_excel_path, thresholds)
# m2_ideal_threshold = thresholds[get_ideal_threshold_index(m2_roc_tpr, m2_roc_fpr)]
# plot_bar_chart(m1_excel_path, up_to_m2_excel_path, m1_threshold, m2_ideal_threshold)
#
# PLOT PATTERN HIGHLIGHTS
# vid = datasets_path + "5 second vids m1 m2 test/SPECIES/VID_NAME.mp4"
# species = SPECIES
# plot_flight_pattern_highlights(vid, species)
