import os
import time

import cv2 as cv
import numpy as np

from ellipse_detector import EllipseDetector
from ellipse_fit_evaluation import evaluate_ellipse_fit
from path_provider import PathProvider
from results_writer import ResultsWriter


def load_image(img_path):
    return cv.imread(img_path, cv.IMREAD_ANYDEPTH)


def image_equalization_to8bit(img):
    return (cv.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX) / 256).astype('uint8')


def print_results_info(results):# print statistics info
    max_score = [0, 4, 6, 8, 12, 15, 20]  # starting 0 cos categories 1-indexed
    total = 0
    for key, value in results.items():
        result = round(sum(value), 1)
        print('Category {0}: {1}/{2}\t{3}%\t\t{4}'.format(key, result, len(value), round(result / len(value) * 100), [round(x, 3) for x in value]))
        print('Score {0}:\t{1}/{2}'.format(key, round(result / len(value) * max_score[key], 1), max_score[key]))
        total += result / len(value) * max_score[key]

    print('\nTotal score: {}\t{}%'.format(round(total, 1), round(total / sum(max_score) * 100, 1)))


def is_hist_mono(hist):
    global_max = []
    # direction 0 -> fall, 1 -> climb
    if hist[1] > hist[0]:
        direction = 1
        new_max = hist[1]
        global_min = hist[0]
    else:
        direction = 1
        new_max = hist[0]
        global_min = 0

    old_value = hist[1]

    for i, hist_bin in enumerate(hist[2:]):
        if hist_bin > new_max and (direction == 1):
            new_max = hist_bin
        elif (hist_bin * 1.2) < new_max and (direction == 1):
            if global_min * 1.3 < new_max:
                global_max.append(new_max)
            new_max = hist_bin
            direction = 0
        elif hist_bin < new_max and (direction == 1):
            pass
        elif hist_bin > new_max and (direction == 0):
            global_min = old_value
            new_max = hist_bin
            direction = 1

        elif hist_bin < new_max and (direction == 0):
            pass

        old_value = hist_bin

    if hist[-1] * 1.5 > global_min and hist[-1] > 800:
        global_max.append("aaaa" + str(global_min) + ", " + str(hist[-1]))

    return len(global_max) != 2


def image_preprocessing(img):
    """
    Does needed img pre-processing.
    :param img:
    :return:
    """
    # PUT IMAGE PRE-PROCESSING HERE:
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))

    dst_morph = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    dst = dst_morph
    dst = cv.GaussianBlur(img, (13, 13), sigmaX=0, sigmaY=0)
    dst = cv.GaussianBlur(dst, (31, 31), sigmaX=0, sigmaY=0)  # 2 times for better blur of peaks
    _, dst = cv.threshold(dst, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    dst = cv.dilate(dst, (15, 15), iterations=5)
    dst = cv.erode(dst, (15, 15), iterations=3)
    dst = cv.bitwise_and(dst, dst_morph)
    return dst


path_provider = PathProvider()
ellipse_detector = EllipseDetector()
results_writer = ResultsWriter('results.csv')

# main
results = {}
paths_dict = path_provider.get_paths_dict()
for category in range(1, 7):
    for path in paths_dict[category]:
        # load image
        image_name = os.path.basename(path)
        image = load_image(path)

        # PROCESSING
        start_time = time.time()  # TODO implement process pool? Much faster program run, but probably slower elapsed_time per image...
        # verify if ellipse exists
        image = image_equalization_to8bit(image)
        hist, _ = np.histogram(image.ravel(), 30, [0, 256])
        if is_hist_mono(hist):
            ellipse = None
        else:
            image = image_preprocessing(image)
            ellipse = ellipse_detector.detect(image, use_distance_filter=True, get_largest=True)

        end_time = time.time()

        # write results to output
        if ellipse:
            ellipse = {'center': (ellipse[0][0], ellipse[0][1]),
                       'axes': (ellipse[1][1] / 2, ellipse[1][0] / 2),
                       'angle': ellipse[2] + 90}
        results_writer.write_result(image_name, ellipse, (end_time - start_time) * 1000)
        result = evaluate_ellipse_fit(image_name, ellipse)
        try:
            results[category].append(result)
        except KeyError:
            results[category] = [result]

print_results_info(results)