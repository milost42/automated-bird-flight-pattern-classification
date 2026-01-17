import numpy as np

"""
Filename: analyse_patterns.py
Author: Mili Ostojic
Date: 09-01-2026
Version: 1.0
Description: Analyse a flight pattern to get its features
"""


def generate_points(flight_pattern, sec_per_frame):
    """Convert labels from flight pattern into numerical points and their corresponding times"""

    # Convert labels into numerical values
    for i in range(len(flight_pattern)):
        if flight_pattern[i] == 'upstroke':
            flight_pattern[i] = 1
        elif flight_pattern[i] == "downstroke":
            flight_pattern[i] = -1
        else:
            flight_pattern[i] = 0

    # Get corresponding time for each point in the flight pattern
    time_points = [0] * len(flight_pattern)
    for i in range(len(time_points)):
        time_points[i] = sec_per_frame * i

    # Calculate total duration of flight pattern
    total_time = sec_per_frame * (len(flight_pattern) - 1)

    return flight_pattern, time_points, total_time


def get_switch(flight, times):
    """Get the points at which the flight pattern switches between upstroke (1) and downstroke (-1)"""

    time_switches = []
    flight_switches = []

    # Define initial point on the flight pattern
    current = flight[0]
    time_switches.append(0)
    flight_switches.append(current)

    # Go through each point in the pattern and determine where the changes are
    for i in range(1, len(flight)):
        if flight[i] != current:
            # Update list of switching times
            time_switches.append(times[i])
            # Update list of switching motion values (i.e. upstroke/downstroke, 1/0)
            flight_switches.append(flight[i])
            current = flight[i]
        else:
            continue

    # Update lists with final point in the pattern
    time_switches.append(times[len(times) - 1])
    flight_switches.append(current)

    return flight_switches, time_switches


def get_diffs(time_switches):
    """Get time differences between switching points"""

    time_diffs = []
    for i in range(len(time_switches) - 1):
        # Calculate time differences between switching points
        difference = time_switches[i + 1] - time_switches[i]
        if difference != 0:
            # Update list of time differences
            time_diffs.append(difference)

    return time_diffs


def get_total_flapping_diffs(time_diffs):
    """Calculate total number of differences and total time of differences between switching points less than 1
    second apart"""

    time_diff_total = 0
    total_flapping_diffs = 0
    for i in range(len(time_diffs)):
        # Add up the total time differences between switching points less than 1 second apart
        if time_diffs[i] < 1.0:
            time_diff_total += time_diffs[i]

            # Calculate total number of differences between switching points less than 1 second apart
            total_flapping_diffs += 1

    return total_flapping_diffs, time_diff_total


def get_flapping_sections(time_diff_total, total_flapping_diffs, time_switches):
    """Get all switching points within each flapping section"""

    # Calculate average time difference between switching points
    time_diff_avg = time_diff_total / total_flapping_diffs

    flapping_sections = []
    section = [time_switches[0]]

    # Cycle through switching points
    for i in range(1, len(time_switches)):

        # If time difference between next switching point and previous one is less than 5 times the average difference
        # or 1 second, whichever is smaller, the switching point is within a flapping section
        if time_switches[i] < time_switches[i - 1] + min(5 * time_diff_avg, 1.0):
            section.append(time_switches[i])
            if i == len(time_switches) - 1:
                flapping_sections.append(section)
        else:
            # If current switching point is not within a flapping section, any current flapping section has ended
            if len(section) > 1:
                # Update list of flapping sections if current flapping section exists
                flapping_sections.append(section)

            # Start a new section
            section = [time_switches[i]]

    return flapping_sections


def get_features(flight_pattern, sec_per_frame):
    """Get features of a flight pattern: average flapping upstroke time, average flapping downstroke time, average time
    spent flapping, average time spent gliding, flapping to gliding ratio, gliding to flapping ratio"""

    # Convert flight pattern to numerical points and corresponding times
    flight_pattern, time_points, total_time = generate_points(flight_pattern, sec_per_frame)

    # Get switching points
    flight_switches, time_switches = get_switch(flight_pattern, time_points)

    # Get time differences between switching points
    time_diffs = get_diffs(time_switches)

    # Get number of time differences that are between switching points less than 1 second apart and total them up
    total_flapping_diffs, time_diff_total = get_total_flapping_diffs(time_diffs)

    # Check if there are time differences less than 1 second apart between switching points
    if total_flapping_diffs > 0:

        # If there are time differences that are between switching points less than 1 second apart, determine which are
        # within flapping sections
        flapping_sections = get_flapping_sections(time_diff_total, total_flapping_diffs, time_switches)

        # Determine the index of the start and end points of each flapping section within the original flight pattern
        flapping_idxs = []
        for section in flapping_sections:
            start_idx = round((section[0] / sec_per_frame))
            end_idx = round((section[len(section) - 1] / sec_per_frame))
            flapping_idxs.append([start_idx, end_idx])

        upstroke_times = []
        downstroke_times = []

        # Cycle through flapping section indexes
        for idx in flapping_idxs:

            # Check end index of the section is the final index of the flight pattern
            if idx[1] == len(flight_pattern) - 1:
                # Extract section from flight pattern that starts at the start index and ends at the end of the pattern
                section = flight_pattern[idx[0]:]
            else:
                # Extract section from flight pattern that starts at the start index and ends at the end index
                section = flight_pattern[idx[0]:idx[1] + 1]

            # Create list of time points for the current flapping section
            section_time_points = [0] * len(section)
            for i in range(len(section_time_points)):
                section_time_points[i] = sec_per_frame * i

            # Get switching points for the points within the current flapping section
            section_flight_switches, section_time_switches = get_switch(section, section_time_points)

            # Cycle through switching points
            for i in range(1, len(section_time_switches)):

                # Check if the previous switching point is an upstroke
                if section_flight_switches[i - 1] == 1:
                    if section_time_switches[i] - section_time_switches[i - 1] != 0:
                        # Update the list of upstroke durations to include the time difference between the current and
                        # previous switching points
                        upstroke_times.append(section_time_switches[i] - section_time_switches[i - 1])

                # Check if the previous switching point is a downstroke
                elif section_flight_switches[i - 1] == -1:
                    if section_time_switches[i] - section_time_switches[i - 1] != 0:
                        # Update the list of downstroke durations to include the time difference between the current
                        # and previous switching points
                        downstroke_times.append(section_time_switches[i] - section_time_switches[i - 1])

        # Check if there is more than one switching point within the flight pattern
        if len(upstroke_times) > 0 and len(downstroke_times) > 0:
            # Calculate average of all upstroke and downstroke times within all flapping sections in the pattern
            upstroke_avg = np.mean(upstroke_times)
            downstroke_avg = np.mean(downstroke_times)
        else:
            # If there are no switching points, there are no flapping sections so the average flapping upstroke and
            # downstroke times are zero
            upstroke_avg = 0
            downstroke_avg = 0

        # get average flapping time and average gliding time
        flapping_times = []
        gliding_times = []

        # Check if there is more than one flapping section
        if len(flapping_sections) > 1:

            # Cycle through flapping sections
            for i in range(len(flapping_sections)):

                # Update list of time spent flapping with the duration of each flapping section
                flapping_times.append(
                    flapping_sections[i][-1] - flapping_sections[i][0])
                if i == 0:
                    # Check if the start of the first flapping section is the start of the flight pattern
                    if flapping_sections[i][0] != 0:
                        # If the start of the first flapping section is not the start of the flight pattern there is
                        # gliding before it, so update the list of time spent gliding with the duration of the gliding
                        # section
                        gliding_times.append(flapping_sections[i][0] - 0)

                    # Update list of time spent gliding with the duration between the current flapping section and the
                    # next flapping section
                    gliding_times.append(flapping_sections[i + 1][0] -
                                         flapping_sections[i][-1])
                elif i == len(flapping_sections) - 1:
                    # Check if the end of the final flapping section is the end of the flight pattern
                    if flapping_sections[i][-1] != total_time:
                        # If the end of the final flapping section is not the end of the flight pattern, it is followed
                        # by a gliding section, so update the list of time spent gliding with the duration of the
                        # gliding section
                        gliding_times.append(total_time - flapping_sections[i][-1])
                else:
                    # Update list of time spent gliding with the duration between the current flapping section and the
                    # next one
                    gliding_times.append(flapping_sections[i + 1][0] -
                                         flapping_sections[i][-1])
        else:
            # Update list of time spent flapping with the duration of the single flapping section
            flapping_times.append(flapping_sections[0][-1] - flapping_sections[0][0])

            # Check if start of flapping section is at the start of the flight pattern
            if flapping_sections[0][0] != 0:
                # If start of flapping section is not at the start of the flight pattern, there is a gliding section
                # before it, so update the list of time spent gliding with the duration of the gliding section
                gliding_times.append(flapping_sections[0][0] - 0)
            # Check if end of flapping section is at the end of the flight pattern
            if flapping_sections[0][-1] != total_time:
                # If the end of the final flapping section is not the end of the flight pattern, it is followed
                # by a gliding section, so update the list of time spent gliding with the duration of the
                # gliding section
                gliding_times.append(total_time - flapping_sections[0][-1])
            else:
                # If neither is true then the whole flight pattern is flapping so update the list of time spent gliding
                # with zero
                if flapping_sections[0][0] == 0:
                    gliding_times.append(0)
    else:
        # If there are no time differences between switching points less than 1 second, the whole pattern is gliding
        flapping_times = [0]
        gliding_times = [total_time]
        upstroke_avg = 0
        downstroke_avg = 0

    # Calculate average flapping time and average gliding time for the whole flight pattern
    average_flapping_time = np.mean(flapping_times)
    average_gliding_time = np.mean(gliding_times)

    # Calculate flapping to gliding ratio
    if average_gliding_time != 0:
        flapping_gliding_ratio = average_flapping_time / average_gliding_time
    else:
        flapping_gliding_ratio = np.nan

    # Calculate gldiing to flapping ratio
    if average_flapping_time != 0:
        gliding_flapping_ratio = average_gliding_time / average_flapping_time
    else:
        gliding_flapping_ratio = np.nan

    # Return list of features
    return [upstroke_avg, downstroke_avg, average_flapping_time, average_gliding_time,
            flapping_gliding_ratio, gliding_flapping_ratio]


