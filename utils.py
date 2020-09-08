import cv2
import time
import numpy as np
from constants import COCO_CLASS_NAMES
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.visualization import draw_box


def draw_fps_on_frame(start_frame_time, img):
    fps = 1 / (time.time() - start_frame_time)
    fps_info_text = f"fps {fps:.3f}"
    cv2.putText(img, fps_info_text, (30, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)


def crop_image_by_box(img, box):
    return img[box[1]:box[3], box[0]:box[2], :]


def crop_image_by_connectedComponentsWithStats(img, stats):
    return img[stats[1]:stats[1] + stats[3], stats[0]:stats[0] + stats[2]]


def is_vehicle(label):
    class_name = COCO_CLASS_NAMES[label]
    return class_name == 'car' or class_name == 'bus' or class_name == 'truck'


def compute_global_box_from_inside_box(relative_box, inside_box):
    inside_box[0] += relative_box[0]
    inside_box[1] += relative_box[1]
    inside_box[2] += relative_box[0]
    inside_box[3] += relative_box[1]

    return inside_box


def create_chars_images_batch(chars_positions, plate_gray_img):
    chars_images = np.array([]).reshape((0, 100, 75, 1))
    for stat in chars_positions:
        char_gray_img = crop_image_by_connectedComponentsWithStats(plate_gray_img, stat)
        # char_bin_img = cv2.adaptiveThreshold(char_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, char_bin_img = cv2.threshold(char_gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        char_bin_img = cv2.resize(char_bin_img, (75, 100))
        char_bin_img = char_bin_img.reshape((1, 100, 75, 1))
        char_bin_img = char_bin_img / 255.0

        chars_images = np.append(chars_images, char_bin_img, axis=0)
    return chars_images


def draw_chars_boxes(chars_positions, draw_bgr_img, plate_box, color):
    for stat in chars_positions:
        draw_box(draw_bgr_img, (plate_box[0] + stat[0], plate_box[1] + stat[1],
                                plate_box[0] + stat[0] + stat[2], plate_box[1] + stat[1] + stat[3]),
                 color=color, thickness=1)

def draw_box_text(img, box, text, colour):
    cv2.putText(img, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 4)
    cv2.putText(img, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, colour, 2)


def visualize_frame(bgr_img):
    bgr_img, _ = resize_image(bgr_img, min_side=800, max_side=1333)
    cv2.imshow("Car detector", bgr_img)


def estimate_images_similarity(img1, img2):
    orb = cv2.ORB_create(edgeThreshold=7, patchSize=7)

    img1 = cv2.Canny(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 100, 200)
    img2 = cv2.Canny(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 100, 200)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None:
        return 0
    matches = bf.match(des1, des2)

    return len(matches)


def find_dominant_colours(img):
    pixels = np.float32(img.reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    _, labels, palette = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)

    return counts, palette
