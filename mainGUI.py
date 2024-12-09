import os
from tkinter import filedialog, Toplevel, Canvas, Scrollbar, Frame
import customtkinter as ctk
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
        self.geometry(f"{1000}x{750}")
        self.configure(fg_color="#f4f4f4")

        # Header Frame
        self.head_frame = ctk.CTkFrame(master=self, corner_radius=10, fg_color="#ffffff")
        self.head_frame.pack(pady=10, padx=10, fill="both")

        self.head_label = ctk.CTkLabel(master=self.head_frame, text="Bone Fracture Detection & Classification",
                                       font=("Roboto", 28, "bold"), text_color="#333333")
        self.head_label.pack(pady=10, padx=10)

        self.info_button = ctk.CTkButton(master=self.head_frame, text="i", width=40, height=40, corner_radius=20,
                                         font=("Roboto", 16, "bold"), fg_color="#2196f3", text_color="#ffffff",
                                         command=self.open_info_window)
        self.info_button.pack(pady=10, padx=10, side="right")

        # Main Frame
        self.main_frame = ctk.CTkFrame(master=self, corner_radius=10, fg_color="#ffffff")
        self.main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.info_label = ctk.CTkLabel(master=self.main_frame,
                                       text="Upload an X-ray image to predict body part and fracture status. Results will be displayed with a heatmap.",
                                       wraplength=800, font=("Roboto", 16), text_color="#555555")
        self.info_label.pack(pady=10, padx=10)

        # Image Display Frame
        self.image_frame = ctk.CTkFrame(master=self.main_frame, fg_color="#f9f9f9", corner_radius=10)
        self.image_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Adding bordered frames for Original Image and Heatmap
        self.original_image_frame = ctk.CTkFrame(master=self.image_frame, fg_color="#ffffff", border_color="#cccccc", border_width=2, corner_radius=10)
        self.original_image_frame.grid(row=0, column=0, padx=10, pady=10)
        self.heatmap_frame = ctk.CTkFrame(master=self.image_frame, fg_color="#ffffff", border_color="#cccccc", border_width=2, corner_radius=10)
        self.heatmap_frame.grid(row=0, column=1, padx=10, pady=10)

        self.original_image_title = ctk.CTkLabel(master=self.original_image_frame, text="Original Image", font=("Roboto", 14, "bold"))
        self.original_image_title.pack(pady=5)

        self.heatmap_title = ctk.CTkLabel(master=self.heatmap_frame, text="Heatmap", font=("Roboto", 14, "bold"))
        self.heatmap_title.pack(pady=5)

        placeholder_path = os.path.join(folder_path, "placeholder.png")
        placeholder_img = Image.open(placeholder_path).resize((256, 256)) if os.path.exists(placeholder_path) else None
        self.placeholder_image = ImageTk.PhotoImage(placeholder_img) if placeholder_img else None

        self.original_image_label = ctk.CTkLabel(master=self.original_image_frame, text="", image=self.placeholder_image)
        self.original_image_label.pack(pady=10)

        self.heatmap_label = ctk.CTkLabel(master=self.heatmap_frame, text="", image=self.placeholder_image)
        self.heatmap_label.pack(pady=10)

        # Buttons Frame
        self.buttons_frame = ctk.CTkFrame(master=self.main_frame, fg_color="#ffffff")
        self.buttons_frame.pack(pady=10, padx=10, fill="x")

        self.upload_btn = ctk.CTkButton(master=self.buttons_frame, text="Upload Image", command=self.upload_image,
                                        font=("Roboto", 14, "bold"), fg_color="#4caf50", text_color="#ffffff")
        self.upload_btn.pack(pady=5, padx=5, side="left")

        self.predict_btn = ctk.CTkButton(master=self.buttons_frame, text="Predict", command=self.predict_gui,
                                         font=("Roboto", 14, "bold"), fg_color="#2196f3", text_color="#ffffff")
        self.predict_btn.pack(pady=5, padx=5, side="left")

        self.reset_btn = ctk.CTkButton(master=self.buttons_frame, text="Reset", command=self.reset_gui,
                                       font=("Roboto", 14, "bold"), fg_color="#f44336", text_color="#ffffff")
        self.reset_btn.pack(pady=5, padx=5, side="left")

        # Results Frame
        self.result_frame = ctk.CTkFrame(master=self.main_frame, fg_color="#ffffff", corner_radius=10)
        self.result_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.result_label = ctk.CTkLabel(master=self.result_frame, text="Results will be displayed here.",
                                         font=("Roboto", 16), text_color="#555555")
        self.result_label.pack(pady=10, padx=10)

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
        self.result_label.configure(text=f"Body Part: {body_part}\nFracture Status: {fracture_status}",
                                     text_color="green")

        # Display heatmap if available
        if heatmap_path:
            heatmap_img = Image.open(heatmap_path).resize((256, 256))
            self.heatmap_image = ImageTk.PhotoImage(heatmap_img)
            self.heatmap_label.configure(image=self.heatmap_image, text="")
        else:
            self.heatmap_label.configure(image=self.placeholder_image, text="")

    def reset_gui(self):
        global filename
        filename = ""
        self.original_image_label.configure(image=self.placeholder_image, text="")
        self.heatmap_label.configure(image=self.placeholder_image, text="")
        self.result_label.configure(text="Results will be displayed here.", text_color="#555555")

    def open_info_window(self):
        info_window = Toplevel(self)
        info_window.title("Instructions")
        info_window.geometry("700x700")

        # Create a scrollable frame for content
        canvas = Canvas(info_window)
        scrollbar = Scrollbar(info_window, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add the rules image or a message
        info_image_path = os.path.join(folder_path, "rules.jpeg")
        if os.path.exists(info_image_path):
            img = Image.open(info_image_path).resize((700, 700))
            info_img = ImageTk.PhotoImage(img)
            info_label = ctk.CTkLabel(master=scrollable_frame, image=info_img, text="")
            info_label.image = info_img
            info_label.pack(pady=10, padx=10)
        else:
            info_label = ctk.CTkLabel(master=scrollable_frame, text="Rules image not found.", font=("Roboto", 14))
            info_label.pack(pady=10, padx=10)

        # Close button
        close_button = ctk.CTkButton(master=scrollable_frame, text="Close", command=info_window.destroy,
                                     font=("Roboto", 14), fg_color="#f44336", text_color="#ffffff")
        close_button.pack(pady=20)

if __name__ == "__main__":
    app = App()
    app.mainloop()
