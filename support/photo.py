"""
OpenCV wrapper for binary images.
with limited Plate Carrée Projection Support
AKA Equirectangular projection
"""
import math
import os
import numpy as np
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2


class Photo:
    def __init__(self):
        self.img = None
        self.height = None
        self.width = None
        self.lat_min, self.lat_max = None, None
        self.lon_min, self.lon_max = None, None
        self.lat, self.lon = None, None

    # def new(self, width, height):
    #     vis = np.zeros((height, width), np.uint8)
    #     self.img = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    #     self.height, self.width = self.img.shape[:2]

    def adopt(self, thing, convert: bool = False):
        self.img = thing.astype('uint8') if not convert else cv2.cvtColor(thing.astype('uint8'), cv2.COLOR_BGR2RGB)
        self.height, self.width = self.img.shape[:2]

    def convert(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    # def convert_a(self, solid=False):
    #     ch = self.img.shape[2]
    #     self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGBA)
    #     if ch < 3:
    #         self.img[:, :, 3] = (0 if not solid else 255)

    def load(self, img_file: str | list, convert: bool = False):
        if isinstance(img_file, list):
            if len(img_file) > 1 or isinstance(img_file[0], list):
                images = []
                for item in img_file:
                    if isinstance(item, list):
                        hoz_images = [cv2.imread(img) for img in item]
                        images.append(cv2.hconcat(hoz_images))
                    else:
                        images.append(cv2.imread(item))
                if len(images) > 1:
                    image = cv2.vconcat(images)
                else:
                    image = images[0]
            else:
                image = cv2.imread(img_file[0])
        else:
            image = None
            if not os.path.isfile(img_file):
                image = cv2.imread(f'{img_file}.png')
            else:
                image = cv2.imread(img_file)
            if image is None:
                print('image not found.')
                exit(-1)
        self.img = image if not convert else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.img.shape[:2]

    def save(self, filename, convert: bool = False):
        if convert:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{filename}.png', self.img)

    def resize(self, sc: float):
        rev = cv2.resize(self.img, (int(sc*self.width), int(sc*self.height)), interpolation=cv2.INTER_NEAREST)
        self.img = rev

    # def crop(self, x0, x1, y0, y1):
    #     cropped = self.img[y0:y1, x0:x1]
    #     self.img = cropped

    def flip(self, code=0):
        # flip code: A flag to specify how to flip the array;
        # 0 means flipping around the x-axis and positive value
        # (for example, 1) means flipping around y-axis.
        # Negative value (for example, -1) means flipping around both axes.
        # Return Value: It returns an image.
        image = cv2.flip(self.img, code)
        self.img = image

    # def set_latlon(self, lat_range, lon_range):
    #     self.height, self.width = self.img.shape[:2]
    #     self.lat_min, self.lat_max = lat_range
    #     self.lon_min, self.lon_max = lon_range
    #     self.lat = np.linspace(self.lat_min, self.lat_max, num=self.height, endpoint=True)
    #     self.lon = np.linspace(self.lon_min, self.lon_max, num=self.width, endpoint=True)

    def show(self, name: str = 'photo', pause: bool = False):
        cv2.imshow(name, self.img)
        if pause:
            cv2.waitKey(0)

    # def draw_grid(self, stroke_col: tuple = (0, 0, 0), stroke_width: int = 1, lat_deg: float = 30., lon_deg: float = 30.):
    #     # color = cv2.cvtColor(stroke_col, cv2.COLOR_BGR2RGB)
    #     img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
    #
    #     color = stroke_col
    #     th_off = stroke_width // 2
    #     # Latitude (hex_layer-S)
    #     if lat_deg > 0:
    #         lat_off = self.lat_min % lat_deg
    #         for lat in np.arange(self.lat_min+lat_off, self.lat_max, lat_deg):
    #             h = self.height - (np.searchsorted(self.lat, lat) % self.height) - 1  # flip self.lat in result such that the last now points to the first.
    #             cv2.line(img, (0, h-th_off), (self.width, h-th_off), color=color, thickness=stroke_width)
    #
    #     # longitude (W-E)
    #     if lon_deg > 0:
    #         lon_off = self.lon_min % lon_deg
    #         for lon in np.arange(self.lon_min+lon_off, self.lon_max, lon_deg):
    #             w = np.searchsorted(self.lon, lon) % self.width
    #             cv2.line(img, (w-th_off, 0), (w-th_off, self.height), color=color, thickness=stroke_width)
    #
    #     self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # def despeckle(self):
    #     vs = [9, 9, 17, 5, 9]
    #     img = self.img
    #     blr = cv2.medianBlur(img, vs[0])  # much better than gaussian in this case.
    #     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #     val = hsv[:, :, 2]
    #     at = cv2.adaptiveThreshold(np.array(255 - val), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, vs[1], vs[2])
    #     ia = np.array(255 - at)  # inversion of adaptiveThreshold of the value.
    #     iv = cv2.adaptiveThreshold(ia, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, vs[3], vs[4])
    #     ib = cv2.subtract(iv, ia)  # use this for add-weighted...
    #     bz = cv2.merge([ib, ib, ib])
    #     self.img = np.where(bz == (0, 0, 0), blr, img)

    def col(self, lat: float, lon: float, flip: bool = True):
        if self.lat is not None and self.lon is not None:
            # given latitude and longitude return it's colour.
            w = np.searchsorted(self.lon, lon) % self.width
            # flip self.lat in result such that the last now points to the first.
            # This is because we look at maps as being 90 at the top, and -90 at the bottom
            if flip:
                h = self.height - (np.searchsorted(self.lat, lat) % self.height) - 1
            else:
                h = np.searchsorted(self.lat, lat) % self.height
            pixel = self.img[h, w]
            return pixel
        else:
            raise RuntimeError('col requires setting latitude and longitude first.')

    # def at(self, x: int, y: int):
    #     if min(self.height-1, max(0, y)) != y or min(self.width-1, max(0, x)) != x:
    #         return 0, 255, 0
    #     b, g, r = self.img[min(self.height-1, max(0, y)), min(self.width-1, max(0, x))]
    #     return int(r), int(g), int(b)


if __name__ == '__main__':
    # h9_test()
    p0 = Photo()
    # p0.load([
    #     ['preparatory/world.topo.bathy.200407.3x21600x21600.B1.png', 'preparatory/world.topo.bathy.200407.3x21600x21600.C1.png']
    # ], False)
    # p0.crop(10800, 21600+10800, 0, 21600)
    # p0.save(f'45EW0_90N_21600')
    #
    #
