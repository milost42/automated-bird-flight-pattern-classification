import os

"""
Filename: generate_patterns.py
Date: 22-01-2026
Version: 1.0.1
Description: Generate the flight patterns for a single clip using the predictions of upstroke/downstroke/non from
Model 2
"""


def generate_patterns(predicted_classes, sec_per_frame):
    """Generate the patterns for a single video from the final set of predicted classes"""

    # Initialise variables
    flight_patterns = []
    first_non_index = 0
    last_non_index = 0
    non_count = 0

    # Add a "start" range to the list of non ranges
    non_ranges = [[0, 0]]

    # Cycle through predicted classes
    for i in range(len(predicted_classes)):

        # Check if predicted class is "non"
        if predicted_classes[i] == 'non':

            # Check if this is the first "non" class in a row
            if non_count == 0:
                # Update index of first "non" class in a row
                first_non_index = i
            else:
                # Update index of last "non" class in a row
                last_non_index = i

            # Increment the number of "non" classes in a row
            non_count += 1
        else:

            # Check if there are more than 5 consecutive "non" classes
            if non_count > 5:
                # Update the non ranges with the current completed one
                non_ranges.append([first_non_index, last_non_index])

            # Reset the number of "non" classes in a row
            non_count = 0

    # Check if there are more than 5 consecutive "non" classes at the end
    if non_count > 5:
        # Update the non ranges with the final completed one
        non_ranges.append([first_non_index, last_non_index])

    # Update the non ranges with a "stop" range at the end
    non_ranges.append([len(predicted_classes) + 1, len(predicted_classes) + 1])
    print(non_ranges)

    # Cycle through ranges (except the "stop" one)
    for i in range(len(non_ranges) - 1):
        # Check if the next non range is at the very start of the video
        # This is needed to deal with the situation in which the first non range other than [0,0] starts at 0
        # While allowing the same code to deal with all other possible non ranges
        if non_ranges[i + 1][0] != 0:
            # If the next non range is not at the very start of the video
            # Update the flight pattern list with a section starting at the first predicted class after the end of the
            # current non range and ending at the first predicted class before the next non range
            flight_patterns.append(predicted_classes[non_ranges[i][1] + 1:non_ranges[i + 1][0]])

    # Cycle through generated flight patterns
    for p in range(len(flight_patterns)):
        # Check if any of the flight patterns are entirely populated with "non" and remove them
        if len(list(set(flight_patterns[p]))) == 1 and list(set(flight_patterns[p]))[0] == "non":
            flight_patterns[p] = []
        else:
            # Remove any remaining "non" labels (less than 5 in a row) from each flight pattern
            final_pattern = []
            for i in range(len(flight_patterns[p])):
                if flight_patterns[p][i] != 'non':
                    final_pattern.append(flight_patterns[p][i])
            # Check if the final pattern is empty
            if len(final_pattern) != 0:
                # Update the flight pattern list with the final pattern and its duration
                flight_patterns[p] = [final_pattern, (len(final_pattern) - 1) * sec_per_frame]

    return flight_patterns

