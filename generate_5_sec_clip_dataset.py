from moviepy.editor import *
import math

"""
Filename: generate_5_sec_clip_dataset.py
Date: 22-01-2026
Version: 1.0.1
Description: Generate 5-second clips from a set of videos separated into species folders
"""


def generate_5_second_clips(address):
    """Generate 5-second clips from an existing set of longer videos"""

    # Address of folder with set of videos
    video_set_folder = "Final video set/"

    # Get all the species folders present
    species_folders = [address + video_set_folder + folder + "/" for folder in os.listdir(address + video_set_folder)]

    # Destination folder for 5 second clips
    clips_folder = address + "5 second vids/"

    # Check if destination folder exists
    clips_folder_exist = os.path.isdir(clips_folder)

    # If destination folder doesn't exist, create it
    if not clips_folder_exist:
        os.mkdir(clips_folder)
    else:
        # If destination folder does exist, remove existing files
        files = os.listdir(clips_folder)
        for file in files:
            file_path = os.path.join(clips_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Cycle through all species sub-folders in video set folder
    for folder in species_folders:

        # Get species name from sub-folder name
        species_name = folder.split("/")[-2]
        print(species_name)

        # Destination address of species sub-folder for 5 second clips
        clips_species_folder = address + "5 second vids/" + species_name

        # Check if destination sub-folder exists
        clips_species_folder_exist = os.path.isdir(clips_species_folder)

        # If destination sub-folder does not exist, create it
        if not clips_species_folder_exist:
            os.mkdir(clips_species_folder)
        else:
            # If destination sub-folder exists, remove existing files
            files = os.listdir(clips_species_folder)
            for file in files:
                file_path = os.path.join(clips_species_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Get list of all videos in the species sub-folder for the video set
        videos = [address + video_set_folder + species_name + "/" + vid for vid in os.listdir(folder)]

        for vid in videos:
            print(vid)

            # Get name of video from video filename
            vid_name = vid.split("/")[-1].split(".")[0]
            print(vid_name)

            # Open video file
            clip = VideoFileClip(vid)

            # Get duration of video
            duration = clip.duration
            print("DURATION:", duration)

            # If video is 5 seconds or longer, split it into 5 second clips
            if duration >= 5:

                # Define window size and shift values
                window_shift = 2
                window_size = 5

                # Define start and end times of new clip
                start = 0
                end = window_size
                count = 0
                print(start, end)

                # Get destination address for the clip
                path = os.path.join(
                    address + "5 second vids/",
                    (folder.split("/")[-2] + "/" + vid_name + "_{:d}" + ".mp4").format(count))
                print(path)

                # Create new clip of window size
                section = clip.subclip(start, end)
                section.write_videofile(path)

                # Increment start and end times by shift value
                start += window_shift
                end += window_shift
                count += 1

                # While the end time is less than the truncated duration of the video, shift the window through the
                # video and generate new 5 second clips
                while end <= math.trunc(duration):

                    # Generate 5-second clip
                    print(start, end)
                    section = clip.subclip(start, end)

                    # Write clip to file
                    path = os.path.join(
                        address + "5 second vids/",
                        (species_name + "/" + vid_name + "_{:d}" + ".mp4").format(count))
                    section.write_videofile(path)

                    # Shift the window
                    start += window_shift
                    end += window_shift
                    count += 1
            else:
                continue


# Change "path/to" to the path to the git folder
dataset_folder_path = "path/to/automated-bird-flight-pattern-classification/Datasets/"
generate_5_second_clips(dataset_folder_path)
