from detecto import core, utils, visualize

"""
Filename: m2.py
Author: Mili Ostojic
Date: 09-01-2026
Version: 1.0
Description: Get predictions of upstroke/downstroke/non for each frame in an input clip using Model 2
"""


def model2_roc(prediction, threshold, action):
    """Get upstroke vs downstroke vs non prediction from all the possible predictions for a single frame"""

    # Break down results into labels, bounding boxes and associated confidence scores
    labels, boxes, scores = prediction

    # Initialise variables
    bird = False
    max_score = 0
    bird_idx = 0

    # Check if any predicted labels exist and cycle through them
    for i in range(len(labels)):

        # If there are any labels (upstroke and/or downstroke), Model 2 has predicted a bird
        bird = True

        # Get label with maximum confidence
        if scores[i] > max_score:
            max_score = scores[i]
            bird_idx = i

    # Check if Model 2 has predicted a bird (i.e. upstroke and/or downstroke)
    if bird:

        # Check if the label with the maximum confidence has a confidence higher than the threshold
        if max_score > threshold:

            # Check if an upstroke/downstroke label is required or just bird
            if action:
                # Update the predicted class with the predicted upstroke/downstroke label
                label = labels[bird_idx]
                predicted_class = label
            else:
                # Update the predicted class with the predicted bird label
                predicted_class = 'bird'
        else:
            # If the maximum confidence doesn't exceed the threshold, assign a "non" label
            predicted_class = 'non'
    else:
        # If no labels exist (upstroke or downstroke), assign a "non" label
        predicted_class = 'non'

    return predicted_class


def model2(frames, bird_frames, threshold):
    """Get upstroke vs downstroke vs non predictions for a set of frames"""

    # Load Model 2
    model = core.Model.load('model2_weights_final.pth',
                            ['upstroke', 'downstroke'])
    torch_model = model.get_internal_model()
    torch_model.eval()

    predicted_classes = []

    # Cycle through frames
    for frame in frames:
        print(frame)

        # Check if the frame was assigned as a bird by M1
        if frame in bird_frames:

            # Get Model 2 prediction for frame
            image = utils.read_image(frame)
            prediction = model.predict(image)

            # Call on Model 2 ROC to avoid repeating code
            predicted_classes.append(model2_roc(prediction, threshold, True))
        else:
            # If the frame is not in the expected folder for a bird prediction, assign a "non" label
            predicted_classes.append('non')

    return predicted_classes
