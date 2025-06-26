import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO

class DuckDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Duck Detector")
        self.root.geometry("800x600")
        self.model = YOLO('yolov8n.pt')
        self.create_widgets()
        self.image_path = None
        self.current_image = None
        self.photo_image = None

    def create_widgets(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=0, column=0, padx=10)

        self.detect_btn = tk.Button(btn_frame, text="Detect Ducks", command=self.detect_ducks)
        self.detect_btn.grid(row=0, column=1, padx=10)

        # self.webcam_btn = tk.Button(btn_frame, text="Webcam Detection", command=self.start_webcam)
        # self.webcam_btn.grid(row=0, column=2, padx=10)

        self.canvas = tk.Canvas(self.root, bg="gray", width=700, height=400)
        self.canvas.pack(pady=10)

        self.result_text = tk.Text(self.root, height=5, width=80)
        self.result_text.pack(pady=10)

    def cv2_to_tkinter(self, img):
        """Convert an OpenCV image to a tkinter compatible image"""
        height, width = img.shape[:2]
        img_bytes = cv2.imencode('.ppm', img)[1].tobytes()
        img_tk = tk.PhotoImage(data=img_bytes, width=width, height=height)
        return img_tk

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if self.image_path:
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.resize_image(img, 700, 400)
            self.current_image = img.copy()
            self.photo_image = self.cv2_to_tkinter(img)
            self.canvas.config(width=self.photo_image.width(), height=self.photo_image.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Loaded image: {self.image_path}\n")

    def detect_ducks(self):
        if self.image_path is None:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please load an image first!\n")
            return

        results = self.model(self.image_path)
        img = self.current_image.copy()
        duck_count = 0

        for detection in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            class_name = results[0].names[int(class_id)]

            if class_name.lower() in ['duck', 'bird']:
                duck_count += 1
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{class_name}: {confidence:.2f}",
                            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.photo_image = self.cv2_to_tkinter(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Found {duck_count} ducks in the image.\n")

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        def update_frame():
            ret, frame = cap.read()
            if ret:
                results = self.model(frame)
                duck_count = 0
                for detection in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = detection
                    class_name = results[0].names[int(class_id)]

                    if class_name.lower() in ['duck', 'bird']:
                        duck_count += 1
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.resize_image(frame, 700, 400)
                self.photo_image = self.cv2_to_tkinter(frame)
                self.canvas.config(width=self.photo_image.width(), height=self.photo_image.height())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Real-time detection: Found {duck_count} ducks\n")
                self.root.after(50, update_frame)
            else:
                cap.release()
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Webcam disconnected\n")
        update_frame()

    def resize_image(self, img, max_width, max_height):
        h, w = img.shape[:2]
        aspect = w / h
        if w > h:
            new_w = min(w, max_width)
            new_h = int(new_w / aspect)
        else:
            new_h = min(h, max_height)
            new_w = int(new_h * aspect)
        return cv2.resize(img, (new_w, new_h))

if __name__ == "__main__":
    root = tk.Tk()
    app = DuckDetectorApp(root)
    root.mainloop()