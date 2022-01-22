import numpy as np
import cv2 as cv
from pathlib import Path
from numpy import ndarray
from matplotlib import pyplot as plt
from .calc import bgr_to_ncc, pixel_likelihood
from .point import Point


class ImageSegment:
    def __init__(self, image: str | ndarray | Path) -> None:
        print(type(image))
        if isinstance(image, Path):
            self.image = cv.imread(str(image))
        elif isinstance(image, str):
            self.image = cv.imread(image)
        elif isinstance(image, ndarray):
            self.image = image
        else:
            raise NotImplementedError

        self.image_histogram = np.squeeze(
            cv.calcHist([self.image.ravel()], [0], None, [256], [0, 255])
        )
        self.cropping = False
        self.selection_rectangle: [Point] = None
        self.reference_point: [Point] = None
        self.mu_r: float = 0
        self.sigma_r: float = 0
        self.mu_g: float = 0
        self.sigma_g: float = 0
        self.bins: int = 0
        self.combined_histogram: ndarray | None = None
        self.roi_histogram: ndarray | None = None
        self.param_out: ndarray | None = None
        self.n_param_out: ndarray | None = None
        self.roi: ndarray | None = None

        self.main()

    def color_picker(self, event: int, x: int, y: int, *args) -> None:
        match event:
            case cv.EVENT_LBUTTONDOWN:
                self.reference_point = [(x, y)]
                self.cropping = True
            case cv.EVENT_LBUTTONUP:
                self.reference_point.append((x, y))
                self.cropping = False
                cv.rectangle(
                    self.image,
                    self.reference_point[0],
                    self.reference_point[1],
                    (0, 255, 0),
                    2,
                )
                cv.imshow("macbeth", self.image)
            case cv.EVENT_MOUSEMOVE:
                if self.cropping:
                    self.selection_rectangle = [(x, y)]

    def get_roi(self) -> None:
        image: ndarray = (self.image / self.image.max()).astype("float32")
        clone = image.copy()
        cv.namedWindow("macbeth", cv.WINDOW_NORMAL)
        cv.setWindowTitle("macbeth", "Select a region of interest")
        if image.shape[0] > image.shape[1]:
            cv.resizeWindow("macbeth", 400, 600)
        else:
            cv.resizeWindow("macbeth", 600, 400)
        cv.setMouseCallback("macbeth", self.color_picker)

        while True:
            if not self.cropping:
                cv.imshow("macbeth", self.image)
            elif self.cropping and self.selection_rectangle:
                rect_copy = image.copy()
                cv.rectangle(
                    rect_copy,
                    self.reference_point[0],
                    self.reference_point[0],
                    (0, 255, 0),
                    1,
                )
                cv.imshow("macbeth", rect_copy)

            key = cv.waitKey(1) & 0xFF
            if key == ord("r"):
                image = clone.copy()
            elif key == ord("c"):
                break

        if len(self.reference_point) == 2:
            self.roi = clone[
                self.reference_point[0][1] : self.reference_point[1][1],
                self.reference_point[0][0] : self.reference_point[1][0],
            ]
            cv.imshow("ROI", self.roi)
            cv.waitKey(0)
        cv.destroyAllWindows()

    def get_chroma_roi(self) -> None:
        i, r, g = bgr_to_ncc(self.roi)
        self.mu_r, self.sigma_r = np.mean(r), np.std(r)
        self.mu_g, self.sigma_g = np.mean(g), np.std(g)

    def get_chroma_img(self) -> None:
        image: ndarray = (self.image / self.image.max()).astype("float32")
        i, r, g = bgr_to_ncc(image)
        pr = pixel_likelihood(r, self.mu_r, self.sigma_r)
        pg = pixel_likelihood(g, self.mu_g, self.sigma_g)
        self.combined_histogram = pr * pg
        self.param_out = self.combined_histogram.copy()

    def get_histogram_roi(self, bins: int = 32, plot_hist: bool = False) -> None:
        i, r, g = bgr_to_ncc(self.roi)
        rint = (r * (bins - 1)).astype("uint8")
        gint = (r * (bins - 1)).astype("uint8")
        rg = np.dstack((rint, gint))
        hist = cv.calcHist([rg], [0, 1], None, [bins, bins], [0, bins - 1, 0, bins - 1])
        if plot_hist:
            plt.figure(figsize=(5, 5))
            cl_hist = np.clip(hist, 0, bins - 1)
            plt.imshow(cl_hist, "gray", origin="lower")
            plt.xlabel("$g$")
            plt.ylabel("$r$")
            plt.grid(0)
            plt.show()
        self.roi_histogram = hist
        self.bins = bins

    def get_histogram_img(self) -> None:
        bins = self.bins
        i, r, g = bgr_to_ncc(self.image)
        rproj = (r * (bins - 1)).astype("uint8")
        gproj = (r * (bins - 1)).astype("uint8")
        proj_array = np.zeros(r.shape)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                proj_array[i, j] = self.roi_histogram[rproj[i, j], gproj[i, j]]
        self.combined_histogram = proj_array
        self.n_param_out = self.combined_histogram.copy()

    def plot_segment(self) -> None:
        fig = plt.figure(figsize=(16, 9))

        ax = fig.add_subplot(121)
        ax.imshow(self.image[:, :, ::-1])
        ax.axis("off")
        ax.grid(0)

        ax = fig.add_subplot(122)
        ax.imshow(self.combined_histogram, "gray")
        ax.axis("off")
        ax.grid(0)

        plt.show()

    def parametric(self):
        self.get_roi()
        self.get_chroma_roi()
        self.get_chroma_img()

    def nonparametric(self, **kwargs):
        if self.reference_point is None:
            self.get_roi()
        self.get_histogram_roi(**kwargs)
        self.get_histogram_img()

    def main(self, savename: str = "", **kwargs):
        self.parametric()
        self.nonparametric(**kwargs)
        if self.image.shape[0] > self.image.shape[1]:
            fig = plt.figure("Segmentation", figsize=(16 / 2, 9 / 2))
        else:
            fig = plt.figure("Segmentation", figsize=(16 / 2 * 2, 9 / 2))

        ax = fig.add_subplot(131)
        ax.imshow(self.image[..., ::-1])
        ax.axis("off")
        ax.grid(0)
        ax.set_title("original")

        ax = fig.add_subplot(132)
        ax.imshow(self.param_out, "gray")
        ax.axis("off")
        ax.grid(0)
        ax.set_title("parametric")

        ax = fig.add_subplot(133)
        ax.imshow(self.n_param_out, "gray")
        ax.axis("off")
        ax.grid(0)
        ax.set_title("non-parametric")

        plt.tight_layout()
        if savename:
            plt.savefig(savename, dpi=300, bbox_inches="tight")
        plt.show()
