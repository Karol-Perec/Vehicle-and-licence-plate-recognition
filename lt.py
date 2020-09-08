import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.exposure import rescale_intensity
from tensorflow.keras import layers, models

# char_model = models.load_model('models/chars/chars_final.h5')

char_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']


def convolve(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output


smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

sharpen2 = np.array((
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]), dtype="int")

laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

org = cv2.imread('./data/plt1.jpg')

# gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
# gray = cv2.filter2D(gray, -1, sharpen)
# bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,15,2)
# _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


kernel = np.ones((3, 3), np.uint8)


# bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
# bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)

# bin = cv2.erode(bin, kernel, iterations=1)
# dilation = cv2.erode(dilation, kernel, iterations=1)
# plt.hist(gray.ravel(),256,[0,256])
# plt.show()

def width_hist(bin):
    hist = np.zeros(bin.shape[1])
    for w in range(bin.shape[1]):
        sum = 0
        for h in range(bin.shape[0]):
            sum = sum + (0 if not bin[h][w] else 1)
        hist[w] = sum
    return hist


def find_characters(plate_img):
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, plate_bin = cv2.threshold(plate_gray, 20, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #plate_bin = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, vehicle_labels, stats, centroids = cv2.connectedComponentsWithStats(plate_bin)
    chars_positions = np.array(
        [stat for stat in stats if (stat[2] < stat[3] and stat[3] > plate_img.shape[0] / 3)])
    if chars_positions.any():
        chars_positions = chars_positions[np.argsort(chars_positions[:, 0])]

    return plate_gray, chars_positions, ret, vehicle_labels, stats, centroids, plate_bin


# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(bin)
# label_hue = np.uint8(179 * labels / np.max(labels))
# blank_ch = 255 * np.ones_like(label_hue)
# labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
# labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
# labeled_img[label_hue == 0] = 0

# stats_filtered = np.array([stat for stat in stats if (stat[2] < stat[3] and stat[3] > org.shape[0]/3)])
# if stats_filtered.any():
#     stats_filtered = stats_filtered[np.argsort(stats_filtered[:, 0])]
#
# for stat in stats_filtered:
#     cv2.rectangle(org, (stat[0], stat[1]), (stat[0]+stat[2], stat[1]+stat[3]), [0, 0, 255], 1, cv2.LINE_AA)
#
# images = np.array([]).reshape((0, 100, 75, 1))
# for stat in stats_filtered:
#     gray_char = gray[stat[1]-5:stat[1]+stat[3]+5, stat[0]-5:stat[0]+stat[2]+5]
#     gray_char = cv2.resize(gray_char, (75, 100))
#     gray_char = gray_char.reshape((1, 100, 75, 1))
#     gray_char = gray_char / 255.0
#
#     images = np.append(images, gray_char, axis=0)
#
# if images.any():
#     new_predictions = char_model.predict_on_batch(images)#np.expand_dims(gray, axis=0)
#     chars = np.apply_along_axis(np.argmax, axis=1, arr=new_predictions)
#     for char in chars:
#         print(char_class_names[char])
#
# pts1 = np.float32([[470, 206], [1479, 198], [32, 1122], [1980, 1125]])
# pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])

plate_gray, chars_positions, ret, labels, stats, centroids, plate_bin = find_characters(org)

label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

points = np.array([[39,15],
                   [0,86],
                   [358,38],
                   [358,112]], dtype='float32')
rectangle = np.zeros((4, 2), dtype='float32')
total = points.sum(axis=1)
rectangle[0] = points[np.argmin(total)]
rectangle[2] = points[np.argmax(total)]
difference = np.diff(points, axis=1)
rectangle[1] = points[np.argmin(difference)]
rectangle[3] = points[np.argmax(difference)]
(a, b, c, d) = rectangle

width1 = np.linalg.norm(c - d)
width2 = np.linalg.norm(b - a)
max_width = max(int(width1), int(width2))

height1 = np.linalg.norm(b - c)
height2 = np.linalg.norm(a - d)
max_height = max(int(height1), int(height2))


print(f'max_width: {max_width}')
print(f'max_height: {max_height}')

vertices = np.array([
                   [0, 0],
                   [max_width - 1, 0],
                   [max_width - 1, max_height - 1],
                   [0, max_height - 1]
                   ], dtype='float32')
M = cv2.getPerspectiveTransform(rectangle, vertices)
perspective = cv2.warpPerspective(src=org, M=M, dsize=(max_width, max_height))


cv2.imshow("labeled_img", labeled_img)
cv2.imshow("org", org)
cv2.imshow("gray", plate_gray)
#cv2.imshow("perspective")

cv2.imwrite("labeled_img.jpg", labeled_img)
cv2.imwrite("labeled_img_bin.jpg", plate_bin)
# cv2.imshow("bin", hist)
# cv2.imshow("dilation", dilation)

cv2.waitKey()
