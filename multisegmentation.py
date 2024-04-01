import cv2
import numpy as np
import matplotlib.pyplot as plt

class MultiSegmentation:
    def __init__(
            self,
            images_path=r'C:\Users\sorek\Downloads\stinkbug.webp',
            min_area=100, #the min area of the object to segmentate
            show_area=True,
            show_height_and_width=True,
            ) :
        """
        Initialize MultiSegmentation object.

        Parameters:
        - images_path (str): Path to the image file.
        - min_area (int): Minimum area of the object to be segmented.
        - show_area (bool): Whether to show area on the segmented objects.
        - show_height_and_width (bool): Whether to show height and width on the segmented objects.
        """
        self.images_path = images_path
        self.min_area = min_area

        self.img = cv2.imread(self.images_path)
        self.mask_list = []

        # Convert the image to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the image to
        # separate the objects from the background
        ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find the contours of the objects in the image
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            tpm_images = self.img.copy()
            if int(area) > self.min_area:

                x, y, w, h = cv2.boundingRect(cnt)
                print("Bounding box: x={}, y={}, width={}, height={}".format(x, y, w, h))
                print(x, y, w, h, area)

                # Create a black image of the same size as the original image
                mask = np.zeros_like(self.img)
                
                # Draw the rectangle on the mask
                cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
                
                # Draw the rectangle on the original image
                cv2.rectangle(tpm_images, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if show_area:
                    cv2.putText(mask, str(area), (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if show_height_and_width:
                    cv2.putText(mask, str("W={}, H={}".format( w, h)), (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                
                self.mask_list.append(cv2.bitwise_and(tpm_images, mask))

    def view_all(self, plot=True):
        """
        Display the original image with bounding boxes around segmented objects.

        Parameters:
        - plot (bool): Whether to plot the image using matplotlib. If False, returns the image.
        """
        img = self.img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the image to
        # separate the objects from the background
        ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find the contours of the objects in the image
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours and calculate the area of each object
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if int(area) > self.min_area:
                # Draw a bounding box around each
                # object and display the area on the image
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, str(area), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if plot:
            try:
                plt.imshow(img)
            except ImportError:
                print("matplotlib is not installed. Unable to plot image.")
        else:
            return img

    def get_mask(self, index=None):
        """
        Get a segmented mask.

        Parameters:
        - index (int): Index of the mask to retrieve.

        Returns:
        - mask (numpy.ndarray): Segmented mask.
        """
        if not isinstance(index, int) or index >= len(self.mask_list):
            raise ValueError(
                f'Invalid index: {index}. Valid indices are between 0 and {len(self.mask_list) - 1}.')
        else:
            return self.mask_list[index]

    def get_all_masks(self):
        """
        Get all segmented masks.

        Returns:
        - mask_list (list): List of segmented masks.
        """
        return self.mask_list


if __name__ == "__main__":
    object = MultiSegmentation(show_area=False, images_path=r'dir', min_area=3000)
    plt.imshow(object.get_mask(0))
    object.view_all()
    plt.show()
