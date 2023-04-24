import cv2
import numpy as np
import subprocess
import sys
import os
try:
    import tkinter as tk
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tkinter'])
    import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model


model_path = "model"
if not os.path.exists(model_path):
    os.makedirs(model_path)
   

filename = "agenet.h5"

path_to_file = os.path.join(model_path,filename)

if not os.path.exists(path_to_file):
    subprocess.check_call(['wget', 'https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/age_only_resnet50_weights.061-3.300-4.410.hdf5', '-O', path_to_file])
model = load_model(path_to_file, compile=False)


# Create a GUI window
window = tk.Tk()


# create a toplevel widget
window.title("Age Estimation")


# get screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set the window size to full screen
window.geometry("%dx%d+0+0" % (screen_width, screen_height))


# Define a function to handle button click events
def predict_age():
    # Show a file dialog to select an image file
    # create a file dialog window
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        initialdir="/",
        filetypes=[("Image files", "*.jpg *.JPG *.jpeg *.png")]
    )

    # Load the image file and preprocess it
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict the age of the person in the image

    age_prob = model.predict(img)[0]
    
    predicted_class = np.argmax(age_prob)
    min_age = 0
    max_age = 100
    estimated_age = int(predicted_class*(max_age-min_age) / (len(age_prob) - 1 ) + min_age)

    img = np.squeeze(img, axis=0)

    # Add text overlay to the image
    text = "Age: " + str(estimated_age)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image with text overlay
    cv2.imshow("Age Estimation Result", img)
    # Wait for the user to close the image window
    while cv2.getWindowProperty("Age Estimation Result", cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all windows and exit the program
    cv2.destroyAllWindows()
    window.destroy()
    sys.exit(0)

# Create a button to select an image file
select_button = tk.Button(window, text="Select Image File", command=predict_age)
select_button.pack()


# Start the GUI event loop
window.mainloop()