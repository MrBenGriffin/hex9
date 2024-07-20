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

    def load(self, img_file: str | list):
        if isinstance(img_file, list):
            if len(img_file) > 1:
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
            image = cv2.imread(f'assets/{img_file}.png')
        self.img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.img.shape[:2]

    def save(self, filename):

        cv2.imwrite(f'output/{filename}.png', self.img)

    def resize(self, sc: float):
        rev = cv2.resize(self.img, (int(sc*self.width), int(sc*self.height)), interpolation=cv2.INTER_NEAREST)
        self.img = rev
        # cv2.resize(self.img, None, amount, cv2.INTER_AREA)

    def set_latlon(self, lat_range, lon_range):
        self.height, self.width = self.img.shape[:2]
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range
        self.lat = np.linspace(self.lat_min, self.lat_max, num=self.height, endpoint=True)
        self.lon = np.linspace(self.lon_min, self.lon_max, num=self.width, endpoint=True)

    def draw_grid(self, stroke_col: tuple = (0, 0, 0), stroke_width: int = 1, lat_deg: float = 30., lon_deg: float = 30.):
        # color = cv2.cvtColor(stroke_col, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

        color = stroke_col
        th_off = stroke_width // 2
        # Latitude (N-S)
        if lat_deg > 0:
            lat_off = self.lat_min % lat_deg
            for lat in np.arange(self.lat_min+lat_off, self.lat_max, lat_deg):
                h = self.height - (np.searchsorted(self.lat, lat) % self.height) - 1  # flip self.lat in result such that the last now points to the first.
                cv2.line(img, (0, h-th_off), (self.width, h-th_off), color=color, thickness=stroke_width)

        # longitude (W-E)
        if lon_deg > 0:
            lon_off = self.lon_min % lon_deg
            for lon in np.arange(self.lon_min+lon_off, self.lon_max, lon_deg):
                w = np.searchsorted(self.lon, lon) % self.width
                cv2.line(img, (w-th_off, 0), (w-th_off, self.height), color=color, thickness=stroke_width)

        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def despeckle(self):
        vs = [9, 9, 17, 5, 9]
        img = self.img
        blr = cv2.medianBlur(img, vs[0])  # much better than gaussian in this case.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        val = hsv[:, :, 2]
        at = cv2.adaptiveThreshold(np.array(255 - val), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, vs[1], vs[2])
        ia = np.array(255 - at)  # inversion of adaptiveThreshold of the value.
        iv = cv2.adaptiveThreshold(ia, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, vs[3], vs[4])
        ib = cv2.subtract(iv, ia)  # use this for add-weighted...
        bz = cv2.merge([ib, ib, ib])
        self.img = np.where(bz == (0, 0, 0), blr, img)

    def col(self, lat: float, lon: float):
        if self.lat:
            # given latitude and longitude return it's colour.
            w = np.searchsorted(self.lon, lon) % self.width
            # flip self.lat in result such that the last now points to the first.
            h = self.height - (np.searchsorted(self.lat, lat) % self.height) - 1
            pixel = self.img[h, w]
            return pixel
        else:
            raise RuntimeError('col requires setting latitude and longitude first.')
