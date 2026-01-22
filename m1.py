import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
Filename: m1.py
Date: 22-01-2026
Version: 1.0.1
Description: Get bird/non predictions for each frame in an input video using Model 1
"""


def model1_roc(results, threshold):
    """Get bird vs non prediction from all the possible predictions for a single frame"""

    # Convert object detection model to binary classifier
    # M1 can fail to identify a bounding box, resulting in an automatic non class unless the threshold is 0.
    if threshold == 0.:
        predicted = "bird"

    # Get prediction
    else:
        # Load Model 1 from TorchHub
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        classes_to_labels = utils.get_coco_object_dictionary()

        # Break down results into bounding boxes, classes and associated confidences
        bboxes, classes, confidences = results

        # Check if any classes have been predicted
        if len(classes) > 0:

            # Initialise variables to be for the "non" class
            predicted = "non"
            bird_bbox = [0, 0, 0, 0]
            bird_class = False

            # Initialise variables for max bird class as zero
            max_bird_idx = 0
            max_bird_confidence = 0

            # Cycle through all predicted classes
            for i in range(len(classes)):

                # Get text label from COCO dictionary
                label = classes_to_labels[classes[i] - 1]
                print(label, confidences[i])

                # Check if the predicted class is "bird" and if it has a higher confidence than previous "bird" labels
                if label == "bird" and confidences[i] > max_bird_confidence:

                    # Update variables
                    max_bird_idx = i
                    max_bird_confidence = confidences[i]
                    bird_class = True

            # Check if at least one of the predicted classes is "bird"
            if bird_class:
                print(max_bird_idx, max_bird_confidence)

                # Check if the maximum confidence for a "bird" class is greater than the threshold
                if max_bird_confidence >= threshold:

                    # If the confidence is greater than the threshold, the final prediction is "bird" with its associated
                    # bounding box
                    predicted = "bird"
                    bird_bbox = bboxes[max_bird_idx]

                    # UNCOMMENT TO PLOT EACH "BIRD" FRAME AND ASSOCIATED BOUNDING BOX
                    # fig, ax = plt.subplots(1)
                    # image = input[0] / 2 + 0.5
                    # ax.imshow(image)
                    # left, bot, right, top = bboxes[max_bird_idx]
                    # x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
                    # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    # ax.add_patch(rect)
                    # ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[max_bird_idx] - 1],
                    #                                   confidences[max_bird_idx] * 100),
                    #         bbox=dict(facecolor='white', alpha=0.5))
                    # plt.show()
                # UNCOMMENT TO PLOT EACH "NON" FRAME
                # else:
                    # fig, ax = plt.subplots(1)
                    # image = input[0] / 2 + 0.5
                    # ax.imshow(image)
                    # plt.show()
            # UNCOMMENT TO PLOT EACH "NON" FRAME
            # else:
            # fig, ax = plt.subplots(1)
            # image = input[0] / 2 + 0.5
            # ax.imshow(image)
            # plt.show()
        else:
            # If no classes are predicted, the final label is "non"
            predicted = "non"

            # Define a bounding box of zeros for a "non" class (it won't be used in subsequent steps)
            bird_bbox = [0, 0, 0, 0]

    # Return the predicted class and associated bounding box
    return predicted, bird_bbox


def model1(frame, threshold):
    """Get bird vs non prediction for a single frame"""

    # Load Model 1 from TorchHub
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

    # Call on Model 1 ROC to avoid repeating code
    prediction, bbox = model1_roc(results, threshold)

    return prediction, bbox


