import numpy as np
import cv2
import time
import sys
from keras_retinanet.utils.visualization import draw_box
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.image import resize_image
from utils import *
from constants import *
from predictions import *
from init import load_models


def main(args=None):
    vehicle_model, plate_model, char_model = load_models()
    setup_gpu(0)
    video = cv2.VideoCapture(sys.argv[1])

    while True:
        i += 1
        start_frame_time = time.time()
        is_successful, img = video.read()
        if not is_successful:
            break
        draw_bgr_img = img.copy()

        # vehicle detection
        vehicle_boxes, vehicle_scores, vehicle_labels = detect_vehicles(img, vehicle_model)
        for vehicle_box, vehicle_score, vehicle_label in zip(vehicle_boxes, vehicle_scores, vehicle_labels):
            if vehicle_score < BETA1 or not is_vehicle(vehicle_label):
                continue
            vehicle_img = crop_image_by_box(draw_bgr_img, vehicle_box)
            draw_box(draw_bgr_img, vehicle_box, color=RED_BGR)

            # plate detection
            plate_box, plate_score = detect_plate(vehicle_img, plate_model)
            if plate_score < BETA2:
                continue
            plate_img = crop_image_by_box(vehicle_img, plate_box)
            plate_box = compute_global_box_from_inside_box(relative_box=vehicle_box, inside_box=plate_box)
            draw_box(draw_bgr_img, plate_box, color=GREEN_BGR)

            # characters detection
            plate_gray_img, chars_positions = find_characters(plate_img)
            chars_images = create_chars_images_batch(chars_positions, plate_gray_img)
            draw_chars_boxes(chars_positions, draw_bgr_img, plate_box, color=BLUE_BGR)
            if not chars_images.any() or chars_images.shape[0] < 5:
                continue

            # characters prediciton
            chars = predict_characters(char_model, chars_images)


            plate_text = ''
            for char in chars:
                plate_text += CHAR_CLASS_NAMES[char]
            print(plate_text)
            draw_box_text(img=draw_bgr_img, box=plate_box, text=plate_text, colour=GREEN_BGR)

        # visualization
        draw_fps_on_frame(start_frame_time, img=draw_bgr_img)
        visualize_frame(draw_bgr_img)

        key = cv2.waitKey(1)
        if key == 81 or key == 113: #press q to exit
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
