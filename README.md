# Two-Way-Sign-Language-
PRE-PROCESSING AND FUTURE EXTRACTION
import os
import cv2
import numpy as np
import random
import pickle
DATADIR = 'dataset'
CATEGORIES = os.listdir(DATADIR)
IMG_SIZE = 50
training_data = []
def create_training_data():
for category in CATEGORIES:
class_num = CATEGORIES.index(category)
path = os.path.join(DATADIR, category)
for img in os.listdir(path):
try:
img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
training_data.append([new_array, class_num])
except Exception as e:
pass
create_training_data()
random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
X.append(features)
y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


TRAINING
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X = X/255.0
y=np.array(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(27))
model.add(Activation("softmax"))
model.compile(loss="sparse_categorical_crossentropy",


optimizer="adam",
metrics=["accuracy"])
y=np.array(y)
phase compared to all the images
history = model.fit(X, y, batch_size=32, epochs=5, validation_split=0.2)
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f'Training accuracy: {train_acc:.4f}')
print("Saved model to disk")
model.save('CNN.model')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
acc=np.array(acc)
val_acc=np.array(val_acc)
loss=np.array(loss)
val_loss=np.array(val_loss)
epochs_range = range(5)
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epochs') # Add x-label
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs') # Add x-label
plt.ylabel('Loss')
plt.show()
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
vgg_y_pred = model.predict_generator(X_test)
y_pred_array=np.array(vgg_y_pred)
print(y_test)


print(y_pred_array)
yt=[]
for xt in y_pred_array:
yt.append(xt.tolist().index(max(xt)))
print(yt)
from sklearn import metrics
acc=(metrics.accuracy_score(yt,y_test)*100)
print("Accuracy is:",acc)
cm1 = metrics.confusion_matrix(yt,y_test)
total1=sum(sum(cm1))
sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, yt))
confusion_mtx = confusion_matrix(y_test, yt)
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx,annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
SIGN TO SPEECH
from textblob import TextBlob
import cv2
import numpy as np
import tensorflow as tf
import os
from playsound import playsound
# Load your trained model
model = tf.keras.models.load_model('CNN.model')
DATADIR = 'dataset'
from gtts import gTTS
import os
import pyttsx3
import os

import time
engine = pyttsx3.init()
engine.setProperty("rate", 150)
CATEGORIES = os.listdir(DATADIR) # Get all subfolder names as categories
IMG_SIZE = 50
ui=0
# Open a webcam feed
cap = cv2.VideoCapture(0)
# Define the dimensions of the box
box_size = 300
# Create a window for displaying the hand image
cv2.namedWindow('Hand Image')
det = ['i have to use washroom','hello how are you ','i have to drink some cup of
water', 'i have a doubt','We need to drawn some blood ','its awesome','is it you','i want
to belt', 'i have a doudt','i will call you','i want a shirt','smile please','i have to stich
my cloths', 'down','i am thirsty','plate','animals','food is not so good','go back','i hate
you','i love you', 'two','three','birds','i will call you later','Knife']
#det=['hi how are you','i dont know','what is your name','who are you','what is
this','where are you','how are you','i am hungry','i am ironman','i love you','i hate
you','i am sick','i am sleeping','i am thirsty','i am in home','thankyou','hi how are
you','i dont know','what is your name','who are you','what is this','where are
you','how are you','i am hungry','i am ironman','i love you','i hate you','i am sick','i
am sleeping','i am thirsty','i am in home','thankyou']
while True:
 ret, frame = cap.read()
 if not ret:
 break
 gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 # Extract the box region for hand
 hand_box = gray_frame[0:box_size, 0:box_size]
 resized_frame = cv2.resize(hand_box, (IMG_SIZE, IMG_SIZE))
 prepared_image = np.array(resized_frame).reshape(-1, IMG_SIZE, IMG_SIZE,
1) / 255.0
 import soundfile as sf
 prediction = model.predict(prepared_image)
 predicted_class = np.argmax(prediction)
 predicted_category = CATEGORIES[predicted_class]
 if "unknown" not in predicted_category:
 m=TextBlob(det[np.argmax(prediction)])
 print(m)

 import win32com.client
 speaker = win32com.client.Dispatch("SAPI.SpVoice")
 speaker.Speak(m)
 #output_file = str(ui) + 'output.mp3'
 #sf.write(output_file,speaker.AudioOutputStream.Buffer,
speaker.AudioOutputStream.Format.FormatTag)
 font = cv2.FONT_HERSHEY_SIMPLEX
 cv2.putText(frame, predicted_category, (10, 30), font, 1, (255, 255, 255), 2,
cv2.LINE_AA)
 # Draw the box on the main frame
 cv2.rectangle(frame, (0, 0), (box_size, box_size), (0, 255, 0), 2)
 # Show the hand image in a separate window
 cv2.imshow('Hand Image', resized_frame)
 cv2.imshow('Hand Sign Recognition', frame)
 if cv2.waitKey(1) & 0xFF == ord('q'):
 break
# Close the 'Hand Image' window
cv2.destroyWindow('Hand Image')
cap.release()
cv2.destroyAllWindows()
SPEECH TO SIGN
import speech_recognition as sr
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
# Function to display the image with Tkinter
def display_image(img_path, meaning):
 window = tk.Tk()
 window.title("Sign Display")
 window.geometry("800x600") # Fixed window size
 # Load and display the image
 img = Image.open(img_path)
 img = img.resize((400, 400), Image.ANTIALIAS)
 img = ImageTk.PhotoImage(img)
 label = Label(window, image=img)
 label.image = img # Keep a reference to avoid garbage collection
 label.pack(pady=10)
 # Display the meaning of the sign
 meaning_label = Label(window,text=f"Meaning: {meaning}", font=("Helvetica",
16), fg="blue")
 meaning_label.pack(pady=10)
 # Close button
 close_button = Button(window, text="Close", command=window.destroy,
font=("Helvetica", 14), bg="red", fg="white")
 close_button.pack(pady=20)
 window.mainloop()
# Speech recognition and mapping to signs
r = sr.Recognizer()
speech = sr.Microphone(device_index=1)
# Mapping phrases to image paths and meanings
signs = {
 "bottle": ("audio/0.jpg", "I need a bottle."),
 "good night": ("audio/1.jpg", "A farewell or bedtime greeting."),
 "want some cup of water": ("audio/2.jpg", "Could you hand me a cup of water?"),
 "i have a doubt": ("audio/3.jpg", "I have a question about this, can you clarify?"),
 "we need to drawn some blood": ("audio/4.jpg", "It’s time to draw some blood,
stay calm."),
 "its awesome": ("audio/5.jpg", "This is incredible! I’m really impressed."),
 "is it you": ("audio/6.jpg", "Is it really you? I almost didn’t recognize you."),
 "i want a belt": ("audio/7.jpg", "Disagreement or refusal."),
 "please": ("audio/8.jpg", "A polite request."),
 "I will call you": ("audio/9.jpg", "I’ll give you a ring."),
 "i want a shirt": ("audio/10.jpg", "I really need a new shirt"),
 "smile please": ("audio/11.jpg", "Hey, smile for me! It’ll make your day better!."),
 "i have to stich my cloths": ("audio/12.jpg", "my clothes need a quick stitch. Better
get the needle!."),
 "down": ("audio/13.jpg", "Things are going down right now! Get ready."),
 "i am thristy": ("audio/14.jpg", "I’m so thirsty. I could really use a drink."),
 "plate": ("audio/15.jpg", "Can you pass me a plate? I’m ready to eat."),
 "animals": ("audio/16.jpg", "They’re so unpredictable and fun to watch."),
 "food is not so good": ("audio/17.jpg", "This food isn’t great. Could use some
improvement."),
 "go back": ("audio/18.jpg", "Wait, we need to go back. I left my things behind"),
 "i hate you": ("audio/19.jpg", "What did I do to upset you?"),
 "i love you": ("audio/20.jpg", "You’re the best."),
 "two": ("audio/21.jpg", "Two."),
 "three": ("audio/22.jpg", "Three"),
 "birds": ("audio/23.jpg", "Look at those birds soaring in the sky, so free."),

 "i will call you later": ("audio/24.jpg", "I’ll call you later, okay? Talk soon!"),
 "knife": ("audio/25.jpg", "Careful with that knife, it’s sharp!"),
}
while True:
 with speech as source:
 print("Say something!")
 r.adjust_for_ambient_noise(source)
 audio = r.listen(source)
 try:
 recog = r.recognize_google(audio, language='en-US')
 print("You said:", recog)
 if recog in signs:
 img_path, meaning = signs[recog]
 display_image(img_path, meaning)
 else:
 print("Phrase not recognized in predefined list.")
 except sr.UnknownValueError:
 print("Google Speech Recognition could not understand audio")
 except sr.RequestError as e:
 print("Could not request results from Google Speech Recognition service;
{0}".format(e))
COMBINE
import cv2
import numpy as np
import tensorflow as tf
import os
import pyttsx3
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
from string import ascii_lowercase
from textblob import TextBlob
import win32com.client
# Initialize pyttsx3 engine for speech output
engine = pyttsx3.init()
engine.setProperty("rate", 150)
# Load pre-trained hand sign recognition model
model = tf.keras.models.load_model('CNN.model')
# Dataset directory and categories for hand signs

DATADIR = 'dataset'
CATEGORIES = os.listdir(DATADIR) # Assumes your dataset is organized by
folder names as categories
IMG_SIZE = 50 # Image size for CNN model input
box_size = 300 # Box size for hand detection area
# Predefined letter-to-index mapping for generating image paths
LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase,
start=1)}
# Predefined list of common phrases for hand sign prediction
det = ['i have to use washroom','hello how are you ','i have to drink some cup of
water',
 'i have a doubt','We need to drawn some blood ','its awesome','is it you','i want
to belt',
 'i have a doudt','i will call you','i want a shirt','smile please','i have to stich my
cloths',
 'down','i am thirsty','plate','animals','food is not so good','go back','i hate you','i
love you',
 'two','three','birds','i will call you later','Knife']
# Function to map text to letter positions
def alphabet_position(text):
 text = text.lower()
 numbers = [LETTERS[character] for character in text if character in LETTERS]
 return numbers # Returns a list of positions as strings
# Function to resize images to a uniform size
def resize_image(image, width=50, height=50):
 return cv2.resize(image, (width, height))
# Speech recognition for Deaf users
# Speech recognition for Deaf users
from PIL import Image, ImageTk
from tkinter import Label, Button
def display_image(img_path, meaning):
 window = tk.Toplevel()
 window.title(meaning)
 # Fixed window size
 # Load and display the image
 from PIL import Image, ImageTk
 img = Image.open(img_path) # Open the image
 img = img.resize((400, 400), Image.Resampling.LANCZOS) # Resize image
 img = ImageTk.PhotoImage(img) # Convert to Tkinter PhotoImage
 label = Label(window, image=img)
 label.image = img # Keep a reference to avoid garbage collection
 label.pack()
 # Display the meaning of the sign
 meaning_label = Label(window, text=f"Meaning: {meaning}",
font=("Helvetica", 16), fg="blue")
 meaning_label.pack(pady=10)
 # Close button
 close_button = Button(window, text="Close", command=window.destroy,
font=("Helvetica", 14), bg="red", fg="white")
 close_button.pack(pady=20)
 window.mainloop()
# Speech recognition and mapping to signs
r = sr.Recognizer()
speech = sr.Microphone(device_index=1)
# Mapping phrases to image paths and meanings
signs = {
 "washroom": ("audio/0.jpg", "I need to freshen up."),
 "good night": ("audio/1.jpg", "A farewell or bedtime greeting."),
 "want some cup of water": ("audio/2.jpg", "Could you hand me a cup of water?"),
 "i have a doubt": ("audio/3.jpg", "I have a question about this, can you clarify?"),
 "we need to drawn some blood": ("audio/4.jpg", "It’s time to draw some blood,
stay calm."),
 "its awesome": ("audio/5.jpg", "This is incredible! I’m really impressed."),
 "is it you": ("audio/6.jpg", "Is it really you? I almost didn’t recognize you."),
 "i want a belt": ("audio/7.jpg", "Disagreement or refusal."),
 "please": ("audio/8.jpg", "A polite request."),
 "I will call you": ("audio/9.jpg", "I’ll give you a ring."),
 "i want a shirt": ("audio/10.jpg", "I really need a new shirt"),
 "smile please": ("audio/11.jpg", "Hey, smile for me! It’ll make your day better!."),
 "i have to stich my cloths": ("audio/12.jpg", "my clothes need a quick stitch. Better
get the needle!."),
 "down": ("audio/13.jpg", "Things are going down right now! Get ready."),
 "i am thristy": ("audio/14.jpg", "I’m so thirsty. I could really use a drink."),
 "plate": ("audio/15.jpg", "Can you pass me a plate? I’m ready to eat."),
 "animals": ("audio/16.jpg", "They’re so unpredictable and fun to watch."),
 "food is not so good": ("audio/17.jpg", "This food isn’t great. Could use some
improvement."),
 "go back": ("audio/18.jpg", "Wait, we need to go back. I left my things behind"),
 "i hate you": ("audio/19.jpg", "What did I do to upset you?"),
 "i love you": ("audio/20.jpg", "You’re the best."),
46
 "two": ("audio/21.jpg", "Two."),
 "three": ("audio/22.jpg", "Three"),
 "birds": ("audio/23.jpg", "Look at those birds soaring in the sky, so free."),
 "i will call you later": ("audio/24.jpg", "I’ll call you later, okay? Talk soon!"),
 "knife": ("audio/25.jpg", "Careful with that knife, it’s sharp!"),
}
def speech_to_image():
 with speech as source:
 print("Say something!")
 r.adjust_for_ambient_noise(source)
 audio = r.listen(source)
 try:
 recog = r.recognize_google(audio, language='en-US')
 print("You said:", recog)
 if recog in signs:
 img_path, meaning = signs[recog]
 display_image(img_path, meaning)
 else:
 print("Phrase not recognized in predefined list.")

 except sr.UnknownValueError:
 print("Google Speech Recognition could not understand audio")
 except sr.RequestError as e:
 print("Could not request results from Google Speech Recognition service;
{0}".format(e))
# Hand sign recognition for Dumb users
def hand_sign_recognition():
 cap = cv2.VideoCapture(0) # Capture video from webcam
 while True:
 ret, frame = cap.read()
 if not ret:
 break
 gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 hand_box = gray_frame[0:box_size, 0:box_size]
 resized_frame = cv2.resize(hand_box, (IMG_SIZE, IMG_SIZE))
 prepared_image = np.array(resized_frame).reshape(-1, IMG_SIZE,
IMG_SIZE, 1) / 255.0
 # Get model prediction
 prediction = model.predict(prepared_image)
 predicted_class = np.argmax(prediction)
 predicted_category = CATEGORIES[predicted_class]
 # Perform text-to-speech if the sign is recognized
 if "unknown" not in predicted_category:
 message = TextBlob(det[predicted_class])
 speaker = win32com.client.Dispatch("SAPI.SpVoice")
 speaker.Speak(str(message))
 # Display prediction on the video feed
 font = cv2.FONT_HERSHEY_SIMPLEX
 cv2.putText(frame, predicted_category, (10, 30), font, 1, (255, 255, 255), 2,
cv2.LINE_AA)
 cv2.rectangle(frame, (0, 0), (box_size, box_size), (0, 255, 0), 2)
 cv2.imshow('Hand Sign Recognition', frame)
 if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit
 break
 cap.release()
 cv2.destroyAllWindows()
# Function to handle user selection (Deaf or Dumb)
def process_selection(choice):
 if choice == "Deaf":
 speech_to_image()
 elif choice == "Dumb":
 hand_sign_recognition()
 else:
 messagebox.showerror("Error", "Invalid choice")
# Tkinter GUI for user interaction
root = tk.Tk()
root.title("Deaf or Dumb Application")
root.geometry("400x200")
# Label for instruction
label = tk.Label(root, text="Are you Deaf or Dumb?", font=("Helvetica", 16))
label.pack(pady=20)
# Buttons for Deaf and Dumb choices
button_deaf = tk.Button(root, text="Deaf", command=lambda:
process_selection("Deaf"), width=10, font=("Helvetica", 14))
button_deaf.pack(side=tk.LEFT, padx=20, pady=20)
button_dumb = tk.Button(root, text="Dumb", command=lambda:
process_selection("Dumb"), width=10, font=("Helvetica", 14))
button_dumb.pack(side=tk.RIGHT, padx=20, pady=20)
root.mainloop()
