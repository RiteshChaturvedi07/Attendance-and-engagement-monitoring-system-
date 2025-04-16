import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2, os, csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime, time
from collections import Counter
from keras.models import model_from_json

# Keep all the existing function definitions the same
# Only changing the UI components

def load_emotion_model():
    with open("facialemotionmodel.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("facialemotionmodel.h5")
    return loaded_model

emotion_model = load_emotion_model()

def predict_emotion(face_roi):
    # Resize the grayscale face ROI to 48x48 and normalize
    resized_face = cv2.resize(face_roi, (48, 48))
    img_array = resized_face.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    predictions = emotion_model.predict(img_array)
    # Original labels from model (7 classes)
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    label = np.argmax(predictions)
    original = emotion_labels[label]
    # Map to only 3 reactions:
    if original in ['happy', 'surprise']:
        return 'happy'
    elif original == 'sad':
        return 'sad'
    else:
        return 'neutral'

def assure_path_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

def contact():
    mess._show(title='Contact us', message="Please contact us on : 'bitdurg.ac.in' ")

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()

def save_pass():
    assure_path_exists("TrainingImageLabel/")
    if os.path.isfile("TrainingImageLabel/psd.txt"):
        with open("TrainingImageLabel/psd.txt", "r") as tf:
            key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas is None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            with open("TrainingImageLabel/psd.txt", "w") as tf:
                tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    op = old.get()
    newp = new.get()
    nnewp = nnew.get()
    if op == key:
        if newp == nnewp:
            with open("TrainingImageLabel/psd.txt", "w") as txf:
                txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()

def change_pass():
    global master
    master = tk.Toplevel(window)
    master.geometry("400x200")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="#f0f0f0")
    
    style = ttk.Style()
    style.configure("TFrame", background="#f0f0f0")
    style.configure("TLabel", background="#f0f0f0", font=('Helvetica', 12))
    style.configure("TEntry", font=('Helvetica', 12))
    style.configure("TButton", font=('Helvetica', 12))
    
    main_frame = ttk.Frame(master, padding="20 20 20 20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(main_frame, text='Enter Old Password:').grid(row=0, column=0, sticky=tk.W, pady=5)
    global old
    old = ttk.Entry(main_frame, width=25, show='*')
    old.grid(row=0, column=1, sticky=tk.W, pady=5)
    
    ttk.Label(main_frame, text='Enter New Password:').grid(row=1, column=0, sticky=tk.W, pady=5)
    global new
    new = ttk.Entry(main_frame, width=25, show='*')
    new.grid(row=1, column=1, sticky=tk.W, pady=5)
    
    ttk.Label(main_frame, text='Confirm New Password:').grid(row=2, column=0, sticky=tk.W, pady=5)
    global nnew
    nnew = ttk.Entry(main_frame, width=25, show='*')
    nnew.grid(row=2, column=1, sticky=tk.W, pady=5)
    
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=3, column=0, columnspan=2, pady=10)
    
    save_btn = ttk.Button(button_frame, text="Save", command=save_pass, style="Accent.TButton")
    save_btn.pack(side=tk.LEFT, padx=5)
    
    cancel_btn = ttk.Button(button_frame, text="Cancel", command=master.destroy)
    cancel_btn.pack(side=tk.LEFT, padx=5)
    
    master.transient(window)
    master.grab_set()
    master.focus_set()

def psw():
    assure_path_exists("TrainingImageLabel/")
    if os.path.isfile("TrainingImageLabel/psd.txt"):
        with open("TrainingImageLabel/psd.txt", "r") as tf:
            key = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas is None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            with open("TrainingImageLabel/psd.txt", "w") as tf:
                tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if password == key:
        TrainImages()
    elif password is None:
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered wrong password')

def clear():
    txt.delete(0, 'end')
    res = "1) Take Images  →  2) Save Profile"
    message1.configure(text=res)

def clear2():
    txt2.delete(0, 'end')
    res = "1) Take Images  →  2) Save Profile"
    message1.configure(text=res)

# Keep all other functions as they are (TakeImages, TrainImages, getImagesAndLabels, TrackImages, load_attendance)
def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    # Read existing IDs from CSV (if any)
    existing_records = []
    if os.path.isfile("StudentDetails/StudentDetails.csv"):
        with open("StudentDetails/StudentDetails.csv", 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    existing_records.append(row[2])
        serial = len(existing_records)
    else:
        with open("StudentDetails/StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
    Id = txt.get()
    name = txt2.get()
    if name.replace(" ", "").isalpha():
        if Id in existing_records:
            answer = mess.askyesno("Confirm Update", f"Student with ID {Id} is already registered. Do you want to update the profile?")
            if not answer:
                return
            else:
                new_records = []
                with open("StudentDetails/StudentDetails.csv", 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 3 and row[2] != Id:
                            new_records.append(row)
                with open("StudentDetails/StudentDetails.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in new_records:
                        writer.writerow(row)
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while True:
            ret, img = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite("TrainingImage/" + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y+h, x:x+w])
                cv2.imshow('Taking Images', img)
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]
        with open("StudentDetails/StudentDetails.csv", 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        message1.configure(text=res)
    else:
        res = "Enter Correct name"
        message.configure(text=res)

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        mess._show(title="Error", message="LBPHFaceRecognizer not available.\nPlease install opencv-contrib-python")
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")
    try:
        recognizer.train(faces, np.array(ID))
    except Exception as e:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Profile Saved Successfully"
    message1.configure(text=res)
    message.configure(text='Total Registrations till now: ' + str(ID[0]))

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        try:
            ID = int(os.path.split(imagePath)[-1].split(".")[1])
        except:
            continue
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    for k in tv.get_children():
        tv.delete(k)
    attendance_dict = {}      # key: id, value: list of predictions
    attendance_record = {}    # key: id, value: (name, date, timeStamp)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        mess._show(title="Error", message="LBPHFaceRecognizer not available.\nPlease install opencv-contrib-python")
        return
    if os.path.isfile("TrainingImageLabel/Trainner.yml"):
        recognizer.read("TrainingImageLabel/Trainner.yml")
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts_val = time.time()
    today_date = datetime.datetime.fromtimestamp(ts_val).strftime('%d-%m-%Y')
    file_path = "Attendance/Attendance_" + today_date + ".csv"
    if os.path.isfile("StudentDetails/StudentDetails.csv"):
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    else:
        mess._show(title='Details Missing', message='Student details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()
        return

    # Run the camera continuously until user presses 'q' or closes the window
    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
            display_text = "Unknown"
            if conf < 70:
                ts_now = time.time()
                date_str = datetime.datetime.fromtimestamp(ts_now).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts_now).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                if len(aa) > 0:
                    name = str(aa[0])
                    id_str = str(serial)
                    predicted = predict_emotion(gray[y:y+h, x:x+w])
                    attendance_dict.setdefault(id_str, []).append(predicted)
                    attendance_record[id_str] = (name, date_str, timeStamp)
                    display_text = name
            cv2.putText(im, display_text, (x, y+h+30), font, 1, (255, 255, 255), 2)
            if conf < 70:
                reaction_text = predict_emotion(gray[y:y+h, x:x+w])
                cv2.putText(im, reaction_text, (x, y+h+60), font, 1, (0, 255, 0), 2)
        cv2.imshow('Taking Attendance', im)
        # Check if 'q' is pressed OR window is closed (X button pressed)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('Taking Attendance', cv2.WND_PROP_VISIBLE) < 1):
            break
    cam.release()
    cv2.destroyAllWindows()

    final_attendance = []
    for id_str, reactions in attendance_dict.items():
        most_common = Counter(reactions).most_common(1)[0][0]
        name, date_str, timeStamp = attendance_record[id_str]
        final_attendance.append([id_str, name, date_str, timeStamp, most_common])
    final_attendance_sorted = sorted(final_attendance, key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

    # Load existing records (if any) and add new records only if they are not duplicates.
    old_records = []
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            old_records = list(reader)

    combined = old_records.copy()
    for new_record in final_attendance_sorted:
        # Check if the ID is already present in the old records
        if not any(old_record[0] == new_record[0] for old_record in old_records):
            combined.append(new_record)

    combined_sorted = sorted(combined, key=lambda row: int(row[0]) if row[0].isdigit() else row[0])
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Name', 'Date', 'Time', 'Emotion'])
        for row in combined_sorted:
            writer.writerow(row)
    tv.delete(*tv.get_children())
    for row in combined_sorted:
        tv.insert('', 'end', values=(row[0], row[1], row[2], row[3], row[4]))

def load_attendance():
    ts_val = time.time()
    today_date = datetime.datetime.fromtimestamp(ts_val).strftime('%d-%m-%Y')
    file_path = "Attendance/Attendance_" + today_date + ".csv"
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            rows = list(reader)
            sorted_rows = sorted(rows, key=lambda row: int(row[0]) if row[0].isdigit() else row[0])
            for line in sorted_rows:
                tv.insert('', 'end', values=(line[0], line[1], line[2], line[3], line[4]))

######################################## USED STUFFS ############################################
global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {'01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
        }

######################################## GUI FRONT-END ###########################################
window = tk.Tk()
window.title("Attendance System with Emotion Recognition")
window.geometry("1280x720")
window.resizable(True, True)
window.configure(background="#f5f5f5")

# Configure styles for ttk widgets
style = ttk.Style()
style.theme_use('clam')  # or 'alt', 'default', 'classic'
style.configure("TFrame", background="#f5f5f5")
style.configure("TLabel", background="#f5f5f5", font=('Helvetica', 11))
style.configure("TButton", font=('Helvetica', 11))
style.configure("Accent.TButton", background="#4CAF50", foreground="white")
style.configure("Warning.TButton", background="#f44336", foreground="white")
style.configure("Info.TButton", background="#2196F3", foreground="white")

# Create the main header
header_frame = ttk.Frame(window)
header_frame.pack(fill=tk.X, pady=(10, 20))

title_label = ttk.Label(header_frame, text="Face Recognition Based Attendance System", 
                         font=('Helvetica', 24, 'bold'))
title_label.pack(side=tk.TOP, pady=10)

date_time_frame = ttk.Frame(header_frame)
date_time_frame.pack(side=tk.TOP)

datef = ttk.Label(date_time_frame, text=day+"-"+mont[month]+"-"+year+"  |  ", 
                  font=('Helvetica', 14))
datef.pack(side=tk.LEFT)

clock = ttk.Label(date_time_frame, font=('Helvetica', 14))
clock.pack(side=tk.LEFT)
tick()

# Create main content frame with two panels
content_frame = ttk.Frame(window)
content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# Left panel - Attendance display
left_panel = ttk.LabelFrame(content_frame, text="Attendance Records")
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

# Create a frame for the treeview
tree_frame = ttk.Frame(left_panel)
tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Create the treeview
tv = ttk.Treeview(tree_frame, columns=("id", "name", "date", "time", "emotion"), show="headings")
tv.heading("id", text="ID")
tv.heading("name", text="NAME")
tv.heading("date", text="DATE")
tv.heading("time", text="TIME")
tv.heading("emotion", text="EMOTION")
tv.column("id", width=80, anchor="center")
tv.column("name", width=150, anchor="center")
tv.column("date", width=100, anchor="center")
tv.column("time", width=100, anchor="center")
tv.column("emotion", width=100, anchor="center")
tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the treeview
scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tv.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
tv.configure(yscrollcommand=scrollbar.set)

# Load attendance data
load_attendance()

# Button frame for left panel
left_button_frame = ttk.Frame(left_panel)
left_button_frame.pack(fill=tk.X, padx=10, pady=10)

trackImg = ttk.Button(left_button_frame, text="Take Attendance", command=TrackImages, style="Info.TButton")
trackImg.pack(side=tk.LEFT, padx=5, pady=10)

refreshBtn = ttk.Button(left_button_frame, text="Refresh Attendance", 
                      command=lambda: [tv.delete(*tv.get_children()), load_attendance()])
refreshBtn.pack(side=tk.LEFT, padx=5, pady=10)

# Right panel - Registration
right_panel = ttk.LabelFrame(content_frame, text="Student Registration")
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

registration_frame = ttk.Frame(right_panel)
registration_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# ID field
id_frame = ttk.Frame(registration_frame)
id_frame.pack(fill=tk.X, pady=10)

lbl = ttk.Label(id_frame, text="Enter ID:")
lbl.pack(side=tk.LEFT, padx=5)

txt = ttk.Entry(id_frame, width=30, font=('Helvetica', 12))
txt.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

clearButton = ttk.Button(id_frame, text="Clear", command=clear, style="Warning.TButton")
clearButton.pack(side=tk.RIGHT, padx=5)

# Name field
name_frame = ttk.Frame(registration_frame)
name_frame.pack(fill=tk.X, pady=10)

lbl2 = ttk.Label(name_frame, text="Enter Name:")
lbl2.pack(side=tk.LEFT, padx=5)

txt2 = ttk.Entry(name_frame, width=30, font=('Helvetica', 12))
txt2.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

clearButton2 = ttk.Button(name_frame, text="Clear", command=clear2, style="Warning.TButton")
clearButton2.pack(side=tk.RIGHT, padx=5)

# Status message
message1 = ttk.Label(registration_frame, text="1) Take Images  →  2) Save Profile")
message1.pack(pady=10)

message = ttk.Label(registration_frame, text="")
message.pack(pady=5)

# Registration buttons
button_frame = ttk.Frame(registration_frame)
button_frame.pack(fill=tk.X, pady=10)

takeImg = ttk.Button(button_frame, text="Take Images", command=TakeImages, style="Info.TButton")
takeImg.pack(fill=tk.X, pady=5)

trainImg = ttk.Button(button_frame, text="Save Profile", command=psw, style="Accent.TButton")
trainImg.pack(fill=tk.X, pady=5)

# Setup count of registrations
res = 0
if os.path.isfile("StudentDetails/StudentDetails.csv"):
    with open("StudentDetails/StudentDetails.csv", 'r') as f:
        reader1 = csv.reader(f)
        for l in reader1:
            res += 1
    res = (res // 2) - 1
else:
    res = 0
message.configure(text='Total Registrations till now: ' + str(res))

# Bottom frame for app controls
bottom_frame = ttk.Frame(window)
bottom_frame.pack(fill=tk.X, pady=10)

quitWindow = ttk.Button(bottom_frame, text="Exit", command=window.destroy, style="Warning.TButton")
quitWindow.pack(side=tk.RIGHT, padx=20)

# Create a menu
menubar = tk.Menu(window)
window.config(menu=menubar)

filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Options", menu=filemenu)
filemenu.add_command(label="Change Password", command=change_pass)
filemenu.add_command(label="Contact Us", command=contact)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=window.destroy)

# Start the main loop
window.mainloop()