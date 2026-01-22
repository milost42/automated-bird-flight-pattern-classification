import os
import sys

import numpy as np
import cv2
import pandas as pd
import torch.cuda
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.markers import MarkerStyle
from detecto import core, utils

from m1 import model1, model1_roc
from m2 import model2, model2_roc
from generate_patterns import generate_patterns
from analyse_patterns import (get_features, generate_points, get_switch, get_diffs, get_total_flapping_diffs,
                              get_flapping_sections)
from get_species import random_forest

"""
Filename: final_object_version.py
Date: 22-01-2026
Version: 1.0.1
Description: Bird object containing all the functions required for analysis of an input clip up to and including species 
identification, as well as all the functions required for testing of each stage of analysis
"""


# Set up all required folders for analysis
temp = os.path.isdir("temp/")
if not temp:
    os.mkdir("temp/")
else:
    files = os.listdir("temp/")
    for file in files:
        file_path = os.path.join("temp/", file)
        if os.path.isfile(file_path):
            os.remove(file_path)
resized_temp = os.path.isdir("resized_temp/")
if not resized_temp:
    os.mkdir("resized_temp/")
else:
    files = os.listdir("resized_temp/")
    for file in files:
        file_path = os.path.join("resized_temp/", file)
        if os.path.isfile(file_path):
            os.remove(file_path)


class Bird:

    def __init__(self, path):
        """Initialise variables"""

        self.vid = path
        self.frames = []
        self.sec_per_frame = 0
        self.resized_frames = []
        self.predicted_classes = []
        self.predicted_bboxes = []
        self.bird_frames = []
        self.motion_predicted = []
        self.flight_patterns = []
        self.predicted_species = ""
        self.predicted_score = 0

    def load_frames(self):
        """Load frames for a single video"""

        # Open video file
        print('Loading frames')
        capture = cv2.VideoCapture(self.vid)
        count = 0

        # Run through the video
        while capture.isOpened():

            # Read frame by frame
            ok, frame = capture.read()

            if ok:
                # Create path for the frame file
                path = os.path.join("temp/", ("frame_{:d}" + ".jpg").format(count))

                # Write frame to temp folder
                cv2.imwrite(path, frame)
                self.frames.append(path)
                count += 1
            else:
                break

            # Calculate seconds per frame
            fps = capture.get(cv2.CAP_PROP_FPS)
            self.sec_per_frame = 1 / fps

        # Exit if the video does not exist or could not be opened
        if not capture.isOpened():
            sys.exit("Video cannot be opened")

        # Release the opened video file
        capture.release()
        cv2.destroyAllWindows()

    def resize_frames(self):
        """Resize the loaded frames to 300x300 pixels as required for Model 1"""

        print('Resizing frames')
        for frame in self.frames:
            # Read frame
            img = cv2.imread(frame)

            # Resize to 300x300 pixels
            width, height = 300, 300
            dsize = (width, height)
            output = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

            # Write resized frame to resized_temp folder
            filename = frame.split("/")[-1]
            self.resized_frames.append("resized_temp/" + filename)
            cv2.imwrite("resized_temp/" + filename, output)

    def model1(self, threshold):
        """Get bird vs non predictions and corresponding bounding boxes from Model 1 for all the resized frames"""

        print('Model 1')

        # Cycle through all resized frames
        for i in range(len(self.resized_frames)):
            # Get prediction and bounding box from Model 1
            predicted, bbox = model1(self.resized_frames[i], threshold)
            print(predicted, bbox)

            # Update list of frames assigned as a bird
            if predicted == "bird":
                self.bird_frames.append(self.frames[i])

            # Update predicted classes and bounding boxes
            self.predicted_classes.append(predicted)
            self.predicted_bboxes.append(bbox)

    def model2(self, threshold):
        """Get upstroke vs downstroke vs non predictions for cropped frames from Model 2"""

        # Get predictions from Model 2
        print('Model 2')
        self.motion_predicted = model2(self.bird_frames, self.frames, threshold)

    def flight_pattern(self):
        """Generate flight patterns from the predicted motion and seconds per frame"""

        # Generate flight patterns
        print('Generating flight patterns')
        self.flight_patterns = generate_patterns(self.motion_predicted, self.sec_per_frame)

        # Remove any empty flight patterns
        self.flight_patterns = [x for x in self.flight_patterns if x != []]
        print(self.flight_patterns)

    def predict_species(self):
        """Predict species for single video from generated flight patterns and Model 3"""

        print('Predicting species')
        predicted_labels = []
        no_predictions = []

        # Get prediction for each flight pattern generated from the video
        for pattern, time in self.flight_patterns:

            # Discard any flight patterns that are shorter than 2 seconds
            if time >= 2.0:

                # Get features of the flight pattern
                features = get_features(pattern, self.sec_per_frame)
                print(features)

                # Replace nan features with zero
                for i in range(len(features)):
                    if np.isnan(features[i]):
                        features[i] = 0

                # Get predicted species from Model 3 for the flight pattern
                predicted_label = str(random_forest(features)[0])
                predicted_labels.append([predicted_label, time])
            else:
                no_predictions.append([None, time])

        # Determine final prediction for the video from the predicted species
        if len(no_predictions) != len(self.flight_patterns):

            # Cross-validated precision for each species
            species = [["Red Kite", 0.6634], ["Kestrel", 0.3884], ["Sparrowhawk", 0.4125],
                       ["Black-headed gull", 0.5063]]
            species_scores = []
            predicted_species = []

            for predicted in predicted_labels:
                predicted_species.append(predicted[0])

            # Calculate confidence score for each possible species from the number of flight patterns predicted with
            # that species and the cross-validated precision scores
            for s, precision in species:
                species_scores.append((predicted_species.count(s) / len(predicted_species)) * precision)

            # Determine most likely final species and associated confidence score
            self.predicted_score = max(species_scores)
            self.predicted_species = species[species_scores.index(max(species_scores))][0]
        else:
            # If no flight patterns longer than 2 seconds generated, return no label
            self.predicted_score = 1
            self.predicted_species = "None"

    def get_patterns(self):
        """Get the flight patterns for a single video"""

        self.load_frames()
        self.resize_frames()
        self.model1(0.1)
        self.model2(0.)
        self.flight_pattern()

        return self.flight_patterns, self.sec_per_frame

    def get_features(self):
        """Get features of the flight patterns longer than 2 seconds for a single video"""

        self.get_patterns()
        all_features = []
        for pattern, time in self.flight_patterns:
            if time >= 2.0:
                features = get_features(pattern, self.sec_per_frame)
                print(features)

                for i in range(len(features)):
                    if np.isnan(features[i]):
                        features[i] = 0

                all_features.append(features)

        return all_features, self.sec_per_frame

    def get_species(self, repeat):
        """Get final predicted species for a single video"""

        # If repeating the same video, loading and resizing frames can be skipped
        if repeat:
            self.model1(0.1)
            self.model2(0.)
            self.flight_pattern()
            self.predict_species()
        else:
            self.load_frames()
            self.resize_frames()
            self.model1(0.1)
            self.model2(0.)
            self.flight_pattern()
            self.predict_species()

        return self.predicted_species, self.predicted_score

    def test_model1(self, frames_dataset_excel, thresholds):
        """Get Model 1 predictions for multiple thresholds from an Excel sheet of test frame paths and their
        actual bird vs non labels"""

        print("STARTING TEST")

        # Read the Excel sheet and retrieve the filenames of the test frames
        df = pd.read_excel(frames_dataset_excel)
        self.frames = list(df["Filename"])
        print(self.frames)

        count = 0
        for frame in self.frames:
            img = cv2.imread(frame)

            # Resize frames to 300x300 pixels for Model 1
            width, height = 300, 300
            dsize = (width, height)
            output = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

            # Write resized frames to resized_temp and update list of resized frames
            filename = "frame_" + str(count) + ".jpg"
            self.resized_frames.append("resized_temp/" + filename)
            cv2.imwrite("resized_temp/" + filename, output)

            count += 1

        print("FRAMES RESIZED")

        all_predicted_classes_per_frame = []

        for frame in self.resized_frames:

            predicted_classes_per_frame = []

            # Retrieve Model 1 from TorchHub
            ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
            utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

            ssd_model.to('cuda')
            ssd_model.eval()

            # Get all possible results (bounding boxes, classes, confidences) for the frame
            with torch.no_grad():
                input = [utils.prepare_input(frame)]
                tensor = utils.prepare_tensor(input)
                detections_batch = ssd_model(tensor)

            results = utils.decode_results(detections_batch)[0]

            # Determine bird vs non prediction for each threshold from the predicted label and associated confidence
            # score
            for threshold in thresholds:

                # If threshold is 0 then all frames should be identified as a bird
                # Model 1 is an object detection model with the option of multiple classes including None so here we
                # convert to a binary model of bird or non
                if threshold == 0.:
                    predicted_classes_per_frame.append("bird")
                else:
                    # Update list with predicted classes per threshold for a single frame
                    predicted, bbox = model1_roc(results, threshold)
                    predicted_classes_per_frame.append(predicted)

            # Update 2D array of predictions per threshold per frame
            all_predicted_classes_per_frame.append(predicted_classes_per_frame)

        # Return list of actual classes and 2D array of predictions per frame per threshold
        return list(df["actual"]), np.array(all_predicted_classes_per_frame)

    def test_model2(self, frames_dataset_excel, thresholds, action):
        """Get Model 2 predictions for multiple thresholds from an Excel sheet of test frame paths and their
        actual bird vs non labels"""

        # Retrieve test frame paths from Excel sheet
        print("STARTING TEST")
        df = pd.read_excel(frames_dataset_excel)
        self.frames = list(df["Filename"])

        # Load Model 2
        model = core.Model.load('model2_weights_final.pth',
                                ['upstroke', 'downstroke'])
        torch_model = model.get_internal_model()
        torch_model.eval()

        # Get Model 2 predictions (labels, bounding boxes, score) for each frame
        predictions = []
        for frame in self.frames:
            print(frame)
            image = utils.read_image(frame)
            predictions.append(model.predict(image))

        # Get predicted motion labels per threshold for each frame
        predicted_motions_per_threshold = []

        # Cycle through thresholds
        for threshold in thresholds:
            predicted_motions = []

            if threshold == 0.:
                # If threshold is 0 then all frames should be identified as a bird
                # Model 1 is an object detection model with the option of multiple classes including None so here we
                # convert to a binary model of bird or non
                for i in range(len(predictions)):
                    predicted_motions.append("bird")
            else:
                # Get predictions from Model 2
                for prediction in predictions:
                    predicted_motions.append(model2_roc(prediction, threshold, action))
            predicted_motions_per_threshold.append(predicted_motions)

        # Return list of actual classes and 2D array of predictions per threshold per frame
        return list(df["actual"]), predicted_motions_per_threshold

    def test_up_to_m2(self, frames_dataset_excel, m1_threshold, m2_threshold):
        """Test M1 and M2 in combination for a single M1 threshold and M2 threshold"""

        # Read the filenames of the frames from the Excel sheet
        df = pd.read_excel(frames_dataset_excel)
        self.frames = list(df["Filename"])

        # Resize the frames to 300x300 pixels
        print('Resizing frames')
        for frame in self.frames:
            img = cv2.imread(frame)
            width, height = 300, 300
            dsize = (width, height)
            output = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

            filename = frame.split("/")[-2] + " " + frame.split("/")[-1]
            self.resized_frames.append("resized_temp/" + filename)
            cv2.imwrite("resized_temp/" + filename, output)

        # Get M1 predictions
        self.model1(m1_threshold)

        # Get M2 predictions
        self.model2(m2_threshold)

        return self.motion_predicted, m2_threshold

    def plot_pattern_highlights(self, species, results_path):
        """Generate and plot flight patterns with switching points and flapping sections highlighted"""

        # Generate the flight patterns for the video
        self.load_frames()
        self.resize_frames()
        self.model1(0.1)
        self.model2(0.)
        self.flight_pattern()

        for pattern, time in self.flight_patterns:

            if time >= 2.0:

                # Set parameters for the plot
                plt.rc('font', size=60)
                plt.rc('axes', titlesize=60)

                fig, ax = plt.subplots(figsize=(20, 10))
                plt.rcParams.update({'font.size': 60})

                plt.yticks([-1, 1], ["downstroke", "upstroke"])

                # Get the points for the plot
                flight_points, time_points, total_time = generate_points(pattern, self.sec_per_frame)

                # Get the switching points
                flight_switches, time_switches = get_switch(flight_points, time_points)

                # Get difference in value and time between switching points
                time_diffs = get_diffs(time_switches)

                # Get the total of the time differences between switching points less than 1 second apart
                total_flapping_diffs, time_diff_total = get_total_flapping_diffs(time_diffs)

                # Check if there are any flapping sections
                if total_flapping_diffs > 0:

                    # Get the flapping sections
                    flapping_sections = get_flapping_sections(time_diff_total, total_flapping_diffs, time_switches)

                    # Highlight the flapping sections with a pink background
                    for section in flapping_sections:
                        start_idx = round((section[0] / self.sec_per_frame))
                        end_idx = round((section[len(section) - 1] / self.sec_per_frame))

                        if start_idx > 0 and end_idx < len(flight_points) - 1:
                            ax.add_patch(Rectangle((time_points[start_idx - 1], -1), time_points[end_idx + 1] -
                                                   time_points[start_idx], 2,
                                                   edgecolor='pink', facecolor='pink', fill=True))
                        elif start_idx > 0:
                            ax.add_patch(Rectangle((time_points[start_idx - 1], -1), time_points[end_idx] -
                                                   time_points[start_idx], 2,
                                                   edgecolor='pink', facecolor='pink', fill=True))
                        else:
                            ax.add_patch(Rectangle((time_points[start_idx], -1), time_points[end_idx] -
                                                   time_points[start_idx], 2,
                                                   edgecolor='pink', facecolor='pink', fill=True))

                # Plot the flight pattern
                plt.plot(time_points, flight_points, linewidth=5, zorder=1)

                # Plot the switching points
                plt.plot(time_switches, flight_switches, c='red', linewidth=0, markersize=25,
                         marker=MarkerStyle('*', fillstyle="full"), zorder=2)

                # Add axis labels and title
                plt.xlabel('Time')
                plt.ylabel('Action')
                plt.title(species + ' flight pattern')

                # Save the plot to file
                vid_name = self.vid.split(".")[0].split("/")[-1]
                save_path = results_path + species + " flight pattern " + vid_name + ".png"
                plt.savefig(save_path, bbox_inches="tight")

                # plt.show()
