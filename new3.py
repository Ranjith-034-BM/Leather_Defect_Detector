import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessorGUI:
    def __init__(self, master):
        self.master = master
        master.title('Image Processor')

        # create buttons
        self.browse_button = tk.Button(master, text='Browse', command=self.browse_image)
        self.process_button = tk.Button(master, text='Process', command=self.process_image)

        # create image label
        self.image_label = tk.Label(master)

        # layout buttons and image label
        self.browse_button.grid(row=0, column=0)
        self.process_button.grid(row=0, column=1)
        self.image_label.grid(row=1, column=0, columnspan=2)

        # initialize image file path and processed image
        self.image_file_path = ''
        self.processed_image = None

    def browse_image(self):
        # open file dialog to select an image file
        file_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg;*.png;*.bmp')])

        # update image file path and display image
        if file_path:
            self.image_file_path = file_path
            self.display_image()

    def process_image(self):
        # load the image
        img = cv2.imread(self.image_file_path)

        # convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply a Gaussian blur to the grayscale image
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # detect edges in the blurred image
        edges = cv2.Canny(blurred_img, 100, 200)

        # convert the processed image to RGB format
        self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # display the processed image
        self.display_image()

    def display_image(self):
        # load the image and resize it to fit in the GUI
        img = Image.open(self.image_file_path)
        img = img.resize((500, 500), Image.ANTIALIAS)

        # if a processed image exists, replace the original image with the processed image
        if self.processed_image is not None:
            img = Image.fromarray(self.processed_image)
            img = img.resize((500, 500), Image.ANTIALIAS)

        # convert the image to Tkinter format and display it
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

root = tk.Tk()
app = ImageProcessorGUI(root)
root.mainloop()
