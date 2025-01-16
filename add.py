import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pickle
from tkinter import *
from tkinter.filedialog import askopenfilename 
from tkinter import filedialog
import subprocess 

action_mapping = {
    "keys": "Custom shortcut",
    "text": "Type words",
    "file": "Open folder",
    "website": "Open website",
    "none": "None"
}

try:
    functions = pickle.load(open('functions.dat', 'rb')) 
except:
    functions = [['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None']] 
print(functions)

def saveF():
    global functions
    print(functions)
    pickle.dump(functions,open("functions.dat", "wb")) 

def website(n):
    print(f"Record action for dropdown {n}")
    def save_url():
        functions[n][0]="website"
        functions[n][1]=f"{url_entry.get()}"
        print(functions)
        saveF() 
        url_win.destroy() 
    url_win = Tk() 
    url_win.title("Enter a link")
 
    url_win.geometry('200x200') 
    url_entry = Entry(url_win)
    url_entry.pack() 
 
    url_button = Button(url_win, command=save_url, text="Finish") 
    url_button.pack()
 
    url_win.mainloop()
 
def openApp(n):
    print(f"Record action for dropdown {n}")
    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_path = folder_path.replace(" ", "\ ") 
        print("Selected folder:", folder_path)
    functions[n][0] = "file"
    functions[n][1] = f"{folder_path}" 
    saveF()
    print(functions)

def text(n):
    #Sub-function to save the text
    def save_text():
        global functions
        functions[n][0]="text"
        functions[n][1]=f"{text_entry.get()}" 
        print(functions)
        saveF() 
        text_win.destroy() 
    text_win = Tk() 
    text_win.title("Please enter your text below")
 
    text_win.geometry('200x200') 
    text_entry = Entry(text_win)
    text_entry.pack() 
 
    text_button = Button(text_win, command=save_text, text="Finish") 
    text_button.pack()
 
    text_win.mainloop()

def start_record(n):
    #Sub-function to save the text
    def save_key():
        global functions
        functions[n][0]="keys"
        functions[n][1]=f"{text_entry.get()}" 
        print(functions)
        saveF() 
        text_win.destroy() 
    text_win = Tk() 
    text_win.title("Please enter your shortcut below")
 
    text_win.geometry('200x200') 
    text_entry = Entry(text_win)
    text_entry.pack() 
 
    text_button = Button(text_win, command=save_key, text="Finish") 
    text_button.pack()
 
    text_win.mainloop()

def cleargesture(n):
    functions[n][0]=None
    functions[n][1]=None
    saveF()

def on_dropdown_change(event, dropdown_index):
    selected_value = event.widget.get()
    button = action_buttons[dropdown_index]

    action_label = action_buttons[dropdown_index]

    if selected_value == "Custom shortcut":
        button.config(text="Record", command=start_record(dropdown_index))
        functions[dropdown_index][0] = "keys"
    elif selected_value == "Type words":
        button.config(text="Add Text", command=text(dropdown_index))
    elif selected_value == "Open folder":
        button.config(text="Open file", command=openApp(dropdown_index))
    elif selected_value == "Open website":
        button.config(text="Open File", command=website(dropdown_index))
    else:
        button.config(text="None", command=cleargesture(dropdown_index))
    
    # Update action label with new content based on action type
    action_content = functions[dropdown_index][1]
    action_label.config(text=action_content)
    root.update()

# Initialize main window
root = tk.Tk()
root.title("Handcut")
window_width = 470
window_height = 630
root.geometry(f"{window_width}x{window_height}")
root.configure(bg="lightblue")
root.resizable(False, False)

# Header frame with logo and title
header_frame = tk.Frame(root, bg="lightblue")
header_frame.pack(pady=20)

# Shared list of dropdown values
dropdown_values = ["None","Custom shortcut", "Type words", "Open folder", "Open file"]
handgestures = [
    "Close", "Swipe right", "Zoom", "Swipeleft", "Cut",
    "Three finger", "Two finger", "Five finger left", "Five finger right", "None"
]

# Add title
title_label = tk.Label(header_frame, text="Handcut", font=("Arial", 24, "bold"), bg="lightblue", fg="black")
title_label.grid(row=0, column=1, padx=10, pady=5)

# Create canvas for scrolling
canvas = tk.Canvas(root)
canvas.pack(side="left", fill="both", expand=True)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)
inner_frame = tk.Frame(canvas, bg="lightblue")
canvas.create_window((0, 0), window=inner_frame, anchor="nw")

# Create list to store action buttons
action_buttons = []

# Create legend, dropdown, and button in two columns
for i in range(10):
    legend_text = handgestures[i]
    
    # Legend label
    legend = tk.Label(inner_frame, text=legend_text, font=("Arial", 14, "bold"), fg="blue", bg="lightblue", bd=1, relief="solid", padx=5, pady=5)
    legend.grid(row=(i % 5)*3, column=i // 5, padx=10, pady=5, sticky="w")
    
    # Dropdown menu with default value based on functions list
    dropdown = ttk.Combobox(inner_frame, values=list(action_mapping.values()), foreground="black")
    action_type = functions[i][0]
    dropdown.set(action_mapping.get(action_type, "None"))
    dropdown.grid(row=(i % 5)*3+1, column=i // 5, padx=10, pady=5, sticky="w")
    dropdown.bind("<<ComboboxSelected>>", lambda event, index=i: on_dropdown_change(event, index))
    
    # Action label to show content of the action
    action_content = functions[i][1]
    action_label = tk.Label(inner_frame, text=action_content, font=("Arial", 12), bg="lightblue")
    action_label.grid(row=(i % 5)*3+2, column=i // 5, padx=10, pady=5, sticky="w")
    action_buttons.append(action_label)

inner_frame.grid_columnconfigure(0, weight=1)
inner_frame.grid_columnconfigure(1, weight=1)
inner_frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

# Mouse scroll binding
def on_mouse_wheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", on_mouse_wheel)

# Run the application
root.mainloop()
