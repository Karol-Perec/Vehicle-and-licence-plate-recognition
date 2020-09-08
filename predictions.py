from keras_retinanet.utils.image import preprocess_image, resize_image
import numpy as np
import cv2
from constants import scene_img_height, scene_img_width


def detect_vehicles(img, model):
    img = preprocess_image(img)
    img, scale = resize_image(img, min_side=scene_img_height,
                              max_side=scene_img_width)  # min_side=800, max_side=1333

    vehicle_boxes, vehicle_scores, vehicle_labels = model.predict_on_batch(np.expand_dims(img, axis=0))
    vehicle_boxes /= scale

    return vehicle_boxes[0].astype(np.int32), vehicle_scores[0], vehicle_labels[0]


def detect_plate(img, model):
    vehicle_image_for_prediction = preprocess_image(img)
    plates_boxes, plates_scores, _ = model.predict_on_batch(
        np.expand_dims(vehicle_image_for_prediction, axis=0))

    return plates_boxes[0][0].astype(np.int32), plates_scores[0][0]


def find_characters(plate_img):
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    #_, plate_bin = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plate_bin = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    ret, vehicle_labels, stats, centroids = cv2.connectedComponentsWithStats(plate_bin)
    chars_positions = np.array(
        [stat for stat in stats if (stat[2] < stat[3] and stat[3] > plate_img.shape[0] / 3)])
    if chars_positions.any():
        chars_positions = chars_positions[np.argsort(chars_positions[:, 0])]

    return plate_gray, chars_positions


def predict_characters(char_model, chars_images):
    chars_predictions = char_model.predict_on_batch(chars_images)
    chars = np.apply_along_axis(np.argmax, axis=1, arr=chars_predictions)
    return chars
