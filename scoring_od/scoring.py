import json
from statistics import mean

def calculate_iou(boxA, boxB):
    # Calculate the Intersection over Union (IoU) of two bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_detection(detected_objects, ground_truth_objects, confidence_threshold, detected_resolution):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Convert detected object coordinates to match ground truth resolution
    detected_objects = convert_coordinates(detected_objects, detected_resolution, (1920, 1080))

    # Match each detected object with ground truth objects
    for detected_obj in detected_objects:
        max_iou = 0
        matched_gt_obj = None
        for gt_obj in ground_truth_objects:
            iou = calculate_iou(detected_obj['box'], gt_obj['box'])
            if iou > max_iou:
                max_iou = iou
                matched_gt_obj = gt_obj

        # Check if the highest IOU is above threshold and the labels match
        if max_iou >= 0.5 and detected_obj['scores'] >= confidence_threshold:
            if matched_gt_obj and detected_obj['label'] == matched_gt_obj['label']:
                true_positives += 1
            else:
                false_positives += 1
        else:
            false_positives += 1

    # Calculate false negatives
    for gt_obj in ground_truth_objects:
        found_match = False
        for detected_obj in detected_objects:
            iou = calculate_iou(detected_obj['box'], gt_obj['box'])
            if iou >= 0.5 and detected_obj['label'] == gt_obj['label']:
                found_match = True
                break
        if not found_match:
            false_negatives += 1

    return true_positives, false_positives, false_negatives

def convert_coordinates(objects, src_resolution, dest_resolution):
    for obj in objects:
        obj['box'] = scale_coordinates(obj['box'], src_resolution, dest_resolution)
    return objects

def scale_coordinates(coordinates, src_resolution, dest_resolution):
    scale_width = dest_resolution[0] / src_resolution[0]
    scale_height = dest_resolution[1] / src_resolution[1]

    scaled_coordinates = [
        int(coordinates[0] * scale_width),
        int(coordinates[1] * scale_height),
        int(coordinates[2] * scale_width),
        int(coordinates[3] * scale_height)
    ]
    return scaled_coordinates

def calculate_f1_score(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

def read_JSON(filename):
    """Reads json file of the given name, parses it and returns the list of dictionaries"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                return data
            else:
                print("Invalid data format in the file:", filename)
                return None
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print("Invalid JSON format in the file:", filename)
        return None

def write_JSON(dictionary, filename):
    existing_data = read_JSON(filename)
    if existing_data is not None:
        # Convert dictionaries to tuples to make them hashable
        existing_data_set = {tuple(d.items()) for d in existing_data}
        new_dict_tuple = tuple(dictionary.items())
        # Check if dictionary already exists
        if new_dict_tuple not in existing_data_set:
            # Append dictionary
            existing_data.append(dictionary)
    else:
        existing_data = [dictionary]

    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)

# Example usage
detected_json = read_JSON("objs_sd_efficientdetd0.json")
ground_truth_json = read_JSON("objs_gt.json")
confidence_threshold = 0.5
detected_res = (854, 480) #(1280, 720) #(1920, 1080)
# print(detected_json[0])
# print(ground_truth_json[0])
f1_scores = []

if len(detected_json) == len(ground_truth_json):
    for detected_objects, ground_truth_objects in zip(detected_json, ground_truth_json):
        true_positives, false_positives, false_negatives = evaluate_detection(list(detected_objects.values())[0], list(ground_truth_objects.values())[0], confidence_threshold, detected_res)
        f1_score = calculate_f1_score(true_positives, false_positives, false_negatives)
        f1_scores.append(f1_score)

data = {"model_name": "EfficientDet-D0", "resolution": "480p", "f1_score": round(mean(f1_scores), 2)}
print(data)

write_JSON(data, "../dataset/xavier-nx/f1_scores/efficientdetd0.json")