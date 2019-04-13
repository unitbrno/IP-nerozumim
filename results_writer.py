#!/usr/bin/env python
import csv


class ResultsWriter:
    def __init__(self, filename):
        self.fd = open(filename, 'w')
        fieldnames = ['filename', 'ellipse_center_x', 'ellipse_center_y', 'ellipse_majoraxis', 'ellipse_minoraxis', 'ellipse_angle', 'elapsed_time']

        self.writer = csv.DictWriter(self.fd, fieldnames)
        self.writer.writeheader()

    def write_result(self, filename, ellipse, elapsed_time):
        """Ellipse format: {'center': (626.76, 494.98), 'axes': (387.96, 381.45), 'angle': 170}"""
        if ellipse is None:
            self.writer.writerow({'filename': filename,
                                  'elapsed_time': elapsed_time})
        else:
            self.writer.writerow({'filename': filename,
                                  'ellipse_center_x': ellipse['center'][0],
                                  'ellipse_center_y': ellipse['center'][1],
                                  'ellipse_majoraxis': ellipse['axes'][0],
                                  'ellipse_minoraxis': ellipse['axes'][1],
                                  'ellipse_angle': ellipse['angle'],
                                  'elapsed_time': elapsed_time})


if __name__ == "__main__":
    rw = ResultsWriter('test.csv')
    rw.write_result('img1.tiff', {'center': (626.76, 494.98), 'axes': (387.96, 381.45), 'angle': 170}, 555)
    rw.write_result('img2.tiff', None, 111)
