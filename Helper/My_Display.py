
import tensorflow as tf
assert tf.__version__.startswith('2')

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path

from PIL import Image

import time
import copy


class My_Display:
    def __init__(self):
        pass

    def make_canvas(self, pic):
        """
        Get the masked list aka all points in the img within the region is
        marked True.
        """
        w = pic.size[0]
        h = pic.size[1]
        # make a canvas with coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T

        return w, h, points

    def get_masks(self, out_lip, in_lip, w, h, points):
        out_poly = Path(out_lip) # make a polygon
        out_grid = out_poly.contains_points(points)
        # now you have a mask with points inside a polygon
        out_mask = out_grid.reshape(h,w)

        in_poly = Path(in_lip) # make a polygon
        in_grid = in_poly.contains_points(points)
        # now you have a mask with points inside a polygon
        in_mask = in_grid.reshape(h,w)

        return out_mask, in_mask

    def set_row(self, in_list, out_list):
        """
        If a masked item is both true in in_list & out_list,
        set item in final_list to False.
        else if a masked item is False in in_list & True in out_list,
        set item  in final_list to True.
        else set everything else in final_list to False.
        """
        final_list = []
        for in_item, out_item in zip(in_list, out_list):
            if in_item == False and out_item == True:
                final_list.append(True)
            else:
                final_list.append(False)

        return final_list

    def set_mask(self, in_mask, out_mask):
        """
        Process every rows/lists in both in_mask, out_mask to get final_mask
        where final_mask is the non_overlapping region between out_mask &
        in_mask (aka 'symmetric_difference').
        """
        final_list = []
        for in_row, out_row in zip(in_mask, out_mask):
            row = self.set_row(in_row, out_row)
            final_list.append(row)

        return final_list

    def get_mask_coords(self, final):
        """
        coords within the masked region.
        """
        coords = []
        for r, arr in enumerate(final):
            for c, bool_item in enumerate(arr):
                if bool_item == True:
                    coord = tuple((c, r))
                    coords.append(coord)

        return coords

    def get_colors(self, img, coords):
        """
        Get original color of all pixels.
        """
        colors = []
        for coord in coords:
            x = round(coord[0])
            y = round(coord[1])
            color = img.getpixel((x, y))
            colors.append(color)

        return colors

    def mean_color(self, colors):
        """
        Get the mean color value of all pixels in non-ovelapping region of the
        2 polygons (out_lip, in_lip).
        """
        R = []
        G = []
        B = []
        for color in colors:
            R.append(color[0])
            G.append(color[1])
            B.append(color[2])
        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)

        return mean_R, mean_G ,mean_B

    def get_color_offset(self, mean_R, mean_G, mean_B, colors):
        """
        Get the offset of each pixel's color value from the mean.
        """
        mean_R = round(mean_R)
        mean_G = round(mean_G)
        mean_B = round(mean_B)

        mean_arr = np.full((len(colors), 3), (mean_R, mean_G, mean_B))

        offsets = colors - mean_arr

        return offsets

    def augment_color(self, pic, coords, offsets, new_val=(0,0,255)):
        """
        Draw each pixel with it's new color & offset.
        """
        #new_val = (255,0,0)
        #new_val = (255,128,128)
        #new_val = (255,64,128)

        #new_val = (64,255,196)

        #new_val = (0,0,255)
        #new_val = (128,64,255)

        for coord, offset in zip(coords, offsets):
            x = coord[0]
            y = coord[1]
            color = new_val + offset
            pic.putpixel((round(x), round(y)), tuple(color))

    def draw_on_lips(self, pic, pic_coords, color):
        """
        Paint the lips region.
        """
        out_lip = pic_coords[58:86]
        in_lip = pic_coords[86:114]

        w, h, points = self.make_canvas(pic)
        out_mask, in_mask = self.get_masks(out_lip, in_lip, w, h, points)
        final = self.set_mask(np.array(in_mask),  np.array(out_mask))
        coords = np.array(self.get_mask_coords(np.array(final)))
        colors = self.get_colors(pic, coords)
        mean_R, mean_G, mean_B = self.mean_color(colors)
        offsets = self.get_color_offset(mean_R, mean_G, mean_B, colors)
        self.augment_color(pic, coords, offsets, color)

        return pic

    def display_test_fine_tune(self, samples,
        saved_model, IMAGE_SIZE,
        fig, rows, cols,
        saved_fig_dir):
        """
        Display test results after fine tuning.
        """

        k=1
        m=2
        for i, batch in enumerate(samples):
            imgs = batch[0]
            coords = batch[1]
            preds = saved_model.predict(imgs)     # prediction
            for j, rec in enumerate(zip(imgs, coords, preds)):
                img = rec[0]
                coord = rec[1]
                pred = rec[2]

                pic = Image.fromarray(img.numpy().astype(np.uint8))
                pic_pred = copy.deepcopy(pic)

                coord_dim = (194,2)
                coord = tf.math.multiply(coord, IMAGE_SIZE)
                coord = tf.reshape(coord, coord_dim)
                pred = tf.math.multiply(pred, IMAGE_SIZE)
                pred = tf.reshape(pred, coord_dim)

                # augment with actual coord.
                pic = self.draw_on_lips(pic, coord, (255,0,0))
                # Augment with predicted coord.
                pic_pred = self.draw_on_lips(pic_pred, pred, (0,0,255))

                fig.add_subplot(rows, cols, k)
                plt.imshow(np.asarray(pic))
                fig.add_subplot(rows, cols, m)
                plt.imshow(np.asarray(pic_pred))

                k=k+2
                m=m+2

        plt.show()
        fig.savefig(saved_fig_dir, bbox_inches='tight')
