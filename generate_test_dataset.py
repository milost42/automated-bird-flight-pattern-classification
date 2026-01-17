import os
from random import randint
from moviepy.editor import *
import shutil
import pandas as pd
import cv2

"""
Filename: generate_test_dataset.py
Author: Mili Ostojic
Date: 09-01-2026
Version: 1.0
Description: Choose four 5-second clips per species from the balanced dataset after manually selecting clips that 
contain frames with no bird visible
"""


def load_frames(frame_address, vid, count):
    """Load frames from a video"""

    frames = []

    # Open video file
    capture = cv2.VideoCapture(vid)

    # Run through the video
    while capture.isOpened():

        # Read frame by frame
        ok, frame = capture.read()

        if ok:
            # Create path for the frame file
            path = os.path.join(frame_address, ("frame_{:d}" + ".jpg").format(count))

            # Write frame to temp folder
            cv2.imwrite(path, frame)
            frames.append(path)

            # Increment the count
            count += 1
        else:
            break

        # Calculate seconds per frame
        fps = capture.get(cv2.CAP_PROP_FPS)

    # Exit if the video does not exist or could not be opened
    if not capture.isOpened():
        sys.exit("Video cannot be opened")

    # Release the opened video file
    capture.release()
    cv2.destroyAllWindows()

    return frames, count


def create_or_wipe_folder(address):
    """Create a folder if it doesn't exist or remove all files from it if it does exist"""

    # Check if folder exists
    address_exist = os.path.isdir(address)

    # If folder does not exist, create it
    if not address_exist:
        os.mkdir(address)
    else:
        # If folder does exist, remove all files within the folder
        files = os.listdir(address)
        for file in files:
            file_path = os.path.join(address, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(file_path, " Removed")


def folder_set_up(address):
    """Set up test videos and test frames dataset folders"""

    # Define all dataset folder address
    balanced_clip_address = address + "5 second vids balanced/"
    test_clip_address = address + "5 second vids m1 m2 test/"
    test_frames_address = address + "5 second vids m1 m2 test frames/"

    # Create or wipe the test clips and test frames folders
    create_or_wipe_folder(test_clip_address)
    create_or_wipe_folder(test_frames_address)

    for folder in os.listdir(balanced_clip_address):
        # Create or wipe the species sub-folders within the test clips folder
        create_or_wipe_folder(test_clip_address + folder)

        # Also create a sub-folder in each species folder for the test videos called "non frames present"
        # For user to manually copy in videos that have periods in which the bird is not on screen
        if not os.path.isdir(test_clip_address + folder + "/non frames present/"):
            os.mkdir(test_clip_address + folder + "/non frames present/")


def generate_test_dataset(address, clips_excel_address, test_clips_excel_address):
    """Generate dataset of test clips and dataset of frames from the chosen test clips once user has manually selected
    videos from the balanced dataset with periods in which the bird is not on screen i.e. non frames present"""

    # Define all dataset folder address
    balanced_clip_address = address + "5 second vids balanced/"
    test_clip_address = address + "5 second vids m1 m2 test/"
    test_frames_address = address + "5 second vids m1 m2 test frames/"

    # Read Excel spreadsheet for balanced dataset of 5 second clips
    vid_dataset = pd.read_excel(clips_excel_address)
    vid_dataset_arr = vid_dataset.to_numpy()
    species_sections = []

    # Split balanced dataset (40 clips per species) into species sections
    for i in range(1, int(len(vid_dataset_arr) / 40)):
        array_index = i * 40
        next_array_index = (i + 1) * 40
        if i == 1:
            species_sections.append(vid_dataset_arr[:array_index])
            species_sections.append(vid_dataset_arr[array_index:next_array_index])
        elif i == int(len(vid_dataset_arr) / 40) - 1:
            species_sections.append(vid_dataset_arr[array_index:])
        else:
            species_sections.append(vid_dataset_arr[array_index:next_array_index])

    print(species_sections)

    all_frames_excel = []
    all_vids_excel = []

    # Cycle through all species
    for species_section in species_sections:

        # Retrieve name of species
        species = species_section[0][1]
        print(species)

        # Get the addresses of all the clips for the current species that contain non frames (manually selected by the
        # user and copied into a "non frames present" sub-folder in the species folder
        available_non_vids_addresses = [balanced_clip_address + species + "/" + vid_address for vid_address in
                                        os.listdir(test_clip_address + species + "/" + "non frames present/")]
        available_non_vids_arrays = []
        groups = []

        # Get the row in the balanced dataset Excel sheet that corresponds to each available clip with non frames
        # present
        for i in range(len(available_non_vids_addresses)):
            for j in range(len(species_section)):
                if available_non_vids_addresses[i] == species_section[j][0]:
                    vid_array = species_section[j]
                    group = vid_array[2]

                    available_non_vids_arrays.append(vid_array)
                    # Group number is related to which video the clip originated from
                    groups.append(group)

        print(available_non_vids_arrays)

        # Check if any of the clips with non frames present come from the same original video
        repeated_groups = (len(set(groups)) < len(groups))

        selected_non_vids = []
        selected_non_vids_arrays = []
        selected_non_groups = []

        if repeated_groups:
            # If there are clips that originate from the same video, select at random a clip for each possible
            # original video group
            for i in range(len(set(groups))):

                # Select a random clip and retrieve its corresponding row from the balanced dataset Excel sheet
                random_index = randint(0, len(available_non_vids_arrays) - 1)
                vid_array = available_non_vids_arrays[random_index]

                # Check if the current selection is already in the list and if there is already a clip in the list
                # that originates from the same original video group
                # Keep selecting at random until the above is not true
                while vid_array[2] in selected_non_groups or vid_array[0] in selected_non_vids:
                    # Select a random clip and retrieve its corresponding row from the balanced dataset Excel sheet
                    random_index = randint(0, len(available_non_vids_arrays) - 1)
                    vid_array = available_non_vids_arrays[random_index]

                # Add selected video to the non vids set
                selected_non_vids.append(vid_array[0])
                selected_non_groups.append(vid_array[2])
                selected_non_vids_arrays.append(vid_array)
        else:
            # Add all selected videos to the non vids set
            for vid_array in available_non_vids_arrays:
                selected_non_vids.append(vid_array[0])
                selected_non_groups.append(vid_array[2])
                selected_non_vids_arrays.append(vid_array)

        print(selected_non_vids)

        final_vids = []
        final_vids_groups = []

        # Check if there are less than 4 available clips with non frames present that originate from different videos
        if len(selected_non_vids) < 4:

            for i in range(len(selected_non_vids)):
                final_vids.append(selected_non_vids[i])
                final_vids_groups.append(selected_non_groups[i])

            # Select random clips from the rest of the balanced dataset to have a total of 4 test clips
            for i in range(4 - len(selected_non_vids)):

                # Select a random clip and retrieve its corresponding row from the balanced dataset Excel sheet
                random_index = randint(0, len(species_section) - 1)
                vid_array = species_section[random_index]

                # Check if the current selection is already in the list and if there is already a clip in the list
                # that originates from the same original video group
                # Keep selecting at random until the above is not true
                while vid_array[2] in final_vids_groups or vid_array[0] in final_vids:
                    # Select a random clip and retrieve its corresponding row from the balanced dataset Excel sheet
                    random_index = randint(0, len(species_section) - 1)
                    vid_array = species_section[random_index]

                # Add selected video to the final test set
                final_vids.append(vid_array[0])
                final_vids_groups.append(vid_array[2])

        else:
            # Select 4 clips from the available clips with non frames present that originate from different videos
            for i in range(4):
                # Select a random clip and retrieve its corresponding row from the balanced dataset Excel sheet
                random_index = randint(0, len(selected_non_vids) - 1)
                vid_array = selected_non_vids_arrays[random_index]

                # Check if the current selection is already in the list and if there is already a clip in the list
                # that originates from the same original video group
                # Keep selecting at random until the above is not true
                while vid_array[2] in final_vids_groups or vid_array[0] in final_vids:
                    random_index = randint(0, len(selected_non_vids) - 1)
                    vid_array = selected_non_vids_arrays[random_index]

                # Add selected video to the final test set
                final_vids.append(vid_array[0])
                final_vids_groups.append(vid_array[2])

        print(final_vids)

        count = 0

        # Populate the test clips and test frames datasets with selected clips and their frames
        for vid in final_vids:
            vid_name = vid.split("/")[-1]

            # Copy selected clips into test clips species folder
            print(test_clip_address + species + "/" + vid_name)
            shutil.copy2(vid, test_clip_address + species + "/" + vid_name)

            # Load frames from all the selected clips into test frames species folder
            frames, count = load_frames(test_frames_address + species + "/", vid, count)

            # Update lists of all test clips and test franes
            for frame in frames:
                all_frames_excel.append(frame)
                all_vids_excel.append(vid)

    # Create Excel sheet of all test frames and their corresponding clips
    frames_excel = {"Filename": all_frames_excel, "Video": all_vids_excel}
    print(frames_excel)
    df = pd.DataFrame(frames_excel)
    df.to_excel(test_clips_excel_address, index=False, header=True)


# Change "path/to" to the path to the git folder
address = "path/to/automated-bird-flight-pattern-classification/Datasets/"

# RUN THIS ONCE AND THEN FOLLOW THE BELOW INSTRUCTIONS
# folder_set_up(address)

# MANUALLY SELECT CLIPS FOR EACH SPECIES FROM BALANCED DATASET WITH NON FRAMES PRESENT AND PUT IN "non frames present"
# FOLDER IN EACH SPECIES SUB-FOLDER
# PATH/automated-bird-flight-pattern-classification/Datasets/5 second vids balanced/SPECIES/non frames present/

clips_excel_address = address + "Dataset excel sheets/5 second clips balanced.xlsx"
test_clips_excel_address = address + "Dataset excel sheets/5 second clips test frames.xlsx"

generate_test_dataset(address, clips_excel_address, test_clips_excel_address)

# LABEL EACH FRAME WITH BIRD/NON IN A COLUMN TITLED "Bird actual" AND UPSTROKE/DOWNSTROKE/NON IN A COLUMN TITLED
# "Motion actual"
