import os
from tkinter import filedialog
import customtkinter as ctk
import pyautogui
import pygetwindow
from PIL import ImageTk, Image

from predictions import predict_with_lrp

# global variables

project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = project_folder + '/images/'

filename = ""

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Bone Fracture Detection & Classification")
        self.geometry(f"{1000}x{700}")

        # Header Frame
        self.head_frame = ctk.CTkFrame(master=self)
        self.head_frame.pack(pady=20, padx=20, fill="both")

        self.head_label = ctk.CTkLabel(master=self.head_frame, text="Bone Fracture Detection & Classification",
                                       font=("Roboto", 28))
        self.head_label.pack(pady=10, padx=10)

        # Main Frame
        self.main_frame = ctk.CTkFrame(master=self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Info Label
        self.info_label = ctk.CTkLabel(master=self.main_frame,
                                       text="Upload an X-ray image to predict body part and fracture status. Results will be displayed with a heatmap.",
                                       wraplength=600, font=("Roboto", 16))
        self.info_label.pack(pady=10, padx=10)

        # Image and Heatmap Frames
        self.image_frame = ctk.CTkFrame(master=self.main_frame)
        self.image_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.original_image_label = ctk.CTkLabel(master=self.image_frame, text="Original Image")
        self.original_image_label.grid(row=0, column=0, padx=10, pady=10)

        self.heatmap_label = ctk.CTkLabel(master=self.image_frame, text="Heatmap")
        self.heatmap_label.grid(row=0, column=1, padx=10, pady=10)

        # Buttons
        self.upload_btn = ctk.CTkButton(master=self.main_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)

        self.predict_btn = ctk.CTkButton(master=self.main_frame, text="Predict", command=self.predict_gui)
        self.predict_btn.pack(pady=10)

        # Results Display
        self.result_label = ctk.CTkLabel(master=self.main_frame, text="", font=("Roboto", 16))
        self.result_label.pack(pady=10)

    def upload_image(self):
        global filename
        f_types = [("Image Files", "*.png;*.jpg;*.jpeg")]
        filename = filedialog.askopenfilename(filetypes=f_types, initialdir=project_folder)

        if filename:
            img = Image.open(filename).resize((256, 256))
            self.original_image = ImageTk.PhotoImage(img)
            self.original_image_label.configure(image=self.original_image, text="")

    def predict_gui(self):
        global filename

        if not filename:
            self.result_label.configure(text="Please upload an image first!", text_color="red")
            return

        # Predict body part and fracture status
        body_part, _ = predict_with_lrp(filename, "Parts")
        fracture_status, heatmap_path = predict_with_lrp(filename, body_part)

        # Display results
        self.result_label.configure(text=f"Body Part: {body_part}\nFracture Status: {fracture_status}", text_color="green")

        # Display heatmap if available
        if heatmap_path:
            heatmap_img = Image.open(heatmap_path).resize((256, 256))
            self.heatmap_image = ImageTk.PhotoImage(heatmap_img)
            self.heatmap_label.configure(image=self.heatmap_image, text="")

if __name__ == "__main__":
    app = App()
    app.mainloop()
