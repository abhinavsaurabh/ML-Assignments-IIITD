#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Hog_descp():
    def __init__(self, input_image, cell_block_size=16, hist_size=8):
        self.input_image = input_image
        self.input_image = np.sqrt(input_image / float(np.max(input_image)))
        self.input_image = self.input_image * 255
        self.Cell_block_size = cell_block_size
        self.hist_size = hist_size
        self.ang_unit = 360 / self.hist_size
        assert type(self.hist_size) == int,
        assert type(self.cell_block_size) == int,
        assert type(self.ang_unit) == int,

    def extract(self):
        hght, width = self.input_image.shape
        gradnt_magntde, gradnt_ang = self.global_gradnt()
        gradnt_magntde = abs(gradnt_magntde)
        gradnt_vector_cell_block = np.zeros((hght / self.cell_block_size, width / self.cell_block_size, self.hist_size))
        for i in range(gradnt_vector_cell_block.shape[0]):
            for j in range(gradnt_vector_cell_block.shape[1]):
                cell_magntde = gradnt_magntde[i * self.cell_block_size:(i + 1) * self.cell_block_size,
                                 j * self.cell_block_size:(j + 1) * self.cell_block_size]
                cell_ang = gradnt_ang[i * self.cell_block_size:(i + 1) * self.cell_block_size,
                             j * self.cell_block_size:(j + 1) * self.cell_block_size]
                gradnt_vector_cell_block[i][j] = self.cell_gradnt(cell_magntde, cell_ang)

        hog_image = self.render_gradnt(np.zeros([hght, width]), gradnt_vector_cell_block)
        hog_vector = []
        for i in range(gradnt_vector_cell_block.shape[0] - 1):
            for j in range(gradnt_vector_cell_block.shape[1] - 1):
                block_vector = []
                block_vector.extend(gradnt_vector_cell_block[i][j])
                block_vector.extend(gradnt_vector_cell_block[i][j + 1])
                block_vector.extend(gradnt_vector_cell_block[i + 1][j])
                block_vector.extend(gradnt_vector_cell_block[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magntde = mag(block_vector)
                if magntde != 0:
                    normalize = lambda block_vector, magntde: [element / magntde for element in block_vector]
                    block_vector = normalize(block_vector, magntde)
                hog_vector.append(block_vector)
        return hog_vector, hog_image


    def get_closest_bins(self, gradnt_ang):
        idx = int(gradnt_ang / self.ang_unit)
        mod = gradnt_ang % self.ang_unit
        if idx == self.hist_size:
            return idx - 1, (idx) % self.hist_size, mod
        return idx, (idx + 1) % self.hist_size, mod

    def global_gradnt(self):
        gradnt_values_x = cv2.Sobel(self.input_image, cv2.CV_64F, 1, 0, ksize=5)
        gradnt_values_y = cv2.Sobel(self.input_image, cv2.CV_64F, 0, 1, ksize=5)
        gradnt_magntde = cv2.addWeighted(gradnt_values_x, 0.5, gradnt_values_y, 0.5, 0)
        gradnt_ang = cv2.phase(gradnt_values_x, gradnt_values_y, angInDegrees=True)
        return gradnt_magntde, gradnt_ang

    def render_gradnt(self, image, cell_gradnt):
        cell_width = self.cell_block_size / 2
        max_mag = np.array(cell_gradnt).max()
        for x in range(cell_gradnt.shape[0]):
            for y in range(cell_gradnt.shape[1]):
                cell_grad = cell_gradnt[x][y]
                cell_grad /= max_mag
                ang = 0
                ang_gap = self.ang_unit
                for magntde in cell_grad:
                    ang_rad = math.radians(ang)
                    x1 = int(x * self.cell_block_size + magntde * cell_width * math.cos(ang_rad))
                    y1 = int(y * self.cell_block_size + magntde * cell_width * math.sin(ang_rad))
                    x2 = int(x * self.cell_block_size - magntde * cell_width * math.cos(ang_rad))
                    y2 = int(y * self.cell_block_size - magntde * cell_width * math.sin(ang_rad))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magntde)))
                    ang += ang_gap

    def cell_gradnt(self, cell_magntde, cell_ang):
        orientation_centers = [0] * self.hist_size
        for i in range(cell_magntde.shape[0]):
            for j in range(cell_magntde.shape[1]):
                gradnt_strength = cell_magntde[i][j]
                gradnt_ang = cell_ang[i][j]
                min_ang, max_ang, mod = self.get_closest_bins(gradnt_ang)
                orientation_centers[min_ang] += (gradnt_strength * (1 - (mod / self.ang_unit)))
                orientation_centers[max_ang] += (gradnt_strength * (mod / self.ang_unit))
        return orientation_centers

        return image
