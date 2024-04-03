import cv2
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import numpy as np

class HorizonDetector():
    def __init__(self, model="LiheYoung/depth-anything-small-hf"):
        self.pipe = pipeline(task="depth-estimation", model=model)
        self.img_size = (640,380)

    def load_image(self, filepath):
        self.file = filepath
        self.img = Image.open(self.file)
        self.img.resize(self.img_size)

    def estimate_depth(self, threshold=1, plot=False):
        # inference
        depth = self.pipe(self.img)["depth"]
        # threshold depth 
        (T, res) = cv2.threshold(np.array(depth), threshold, 255, cv2.THRESH_BINARY)
        self.mask_image = res.astype("uint8")
        if plot:
            plt.imshow(np.array(depth))
            plt.imshow(res)

    def get_image_with_skyline(self):
        # Draw the horizon contour directly on the color image
        img_np = np.array(self.img)
        img_line = cv2.polylines(img_np.astype("uint8"), [self.horizon_contour], False, (255, 0, 0), thickness=2)  # Red color for visibility
        return img_line


    def extract_horizon(self, plot=False):
        # extract edges
        edges = cv2.Canny(self.mask_image, 50, 100) 

        # find contours in the binary mask
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # aggregate contours
        self.horizon_contour = np.empty((1,2), dtype='int32')
        for c in contours:
            self.horizon_contour = np.append(self.horizon_contour, c.reshape(c.shape[0], -1), axis=0)
        self.horizon_contour = np.delete(self.horizon_contour, [0], axis=0)
        self.horizon_contour = self.horizon_contour[np.argsort(self.horizon_contour[:,0])]
        self.horizon_contour = self.horizon_contour.reshape((self.horizon_contour.shape[0], 1,2))

        if plot:
            img_line = self.get_image_with_skyline()
            # Let's display the contour image
            plt.imshow(img_line)
            plt.title('Extracted Horizon Line')
            plt.axis('off')
            plt.show()

    def save_horizon(self, datapath, name):
        np.save(datapath+name+'.npy',self.horizon_contour)


def depths_no_sky(depths):
    '''Remove sky in depth map

    Args:
        depths (np.array): depth map

    Returns:
        np.array: depth map with sky replaced by nan
    '''
    d = depths.copy()
    # filter 0 values to nan (no depth for sky)
    d[d==0] = np.nan
    # keep bottom half of original depth image (in case close objects are set to 0 depth)
    # half = int(depths.shape[0]/2)
    # d[half:,:] = depths[half:,:]
    return d

def compare(images, title):
    '''plot multiple images

    Args:
        images (list of np.array): list ofimages to plot
        title (str): plot title
    '''
    n = len(images)
    fig, axarr = plt.subplots(1, n, figsize=(7*n, 15))
    for i, img in enumerate(images):
        if isinstance(img, np.ndarray):
            axarr[i].imshow(img)
        elif isinstance(img, plt.Figure):
            axarr[i].imshow(img.canvas.renderer.buffer_rgba())
        else:
            raise TypeError(f"Unsupported plot type: {type(img)}")
        axarr[i].axis('off')
    plt.tight_layout()
    fig.suptitle(title, size=20)
    fig.subplots_adjust(top=1.6)

    plt.show()