import os
import time

import cv2 as cv

from ellipse_detector import EllipseDetector
from ellipse_fit_evaluation import evaluate_ellipse_fit
from path_provider import PathProvider
from results_writer import ResultsWriter


def load_image(img_path):
    return cv.imread(img_path, cv.IMREAD_ANYDEPTH)


def image_equalization_to8bit(img):
    return (cv.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX) / 256).astype('uint8')


def processing_f1(img):
    # PUT IMAGE PRE-PROCESSING HERE:

    return img


# init
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

        # fit ellipse
        start_time = time.time()  # TODO don't use process pool - much faster program, but slower elapsed_time per image...
        image = image_equalization_to8bit(image)
        image = processing_f1(image)
        ellipse = ellipse_detector.detect(image, use_distance_filter=True, get_largest=True)
        end_time = time.time()

        # write results to output
        if ellipse:
            ellipse = {'center': (ellipse[0][0], ellipse[0][1]),
                       'axes': (ellipse[1][1] / 2, ellipse[1][0] / 2),
                       'angle': ellipse[2] + 90}
        results_writer.write_result(image_name, ellipse, (end_time - start_time)*1000)
        result = evaluate_ellipse_fit(image_name, ellipse)
        try:
            results[category].append(result)
        except KeyError:
            results[category] = [result]

# print statistics info
max_score = [0, 4, 6, 8, 12, 15, 20]  # starting 0 cos categories 1-indexed
total = 0
for key, value in results.items():
    result = round(sum(value), 1)
    print('Category {0}: {1}/{2}\t{3}%\t\t{4}'.format(key, result, len(value), round(result / len(value) * 100), [round(x, 3) for x in value]))
    print('Score {0}:\t{1}/{2}'.format(key, round(result / len(value) * max_score[key], 1), max_score[key]))
    total += result / len(value) * max_score[key]

print('\nTotal score: {}\t{}%'.format(round(total, 1), round(total / sum(max_score) * 100, 1)))
