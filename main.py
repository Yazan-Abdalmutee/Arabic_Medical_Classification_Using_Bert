import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox
import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

model_name = 'aubmindlab/bert-base-arabertv02'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=8, 
                                                      output_attentions=False, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=100)

id2label = {
    0: 'أمراض الدم',
    1: 'الأورام الخبيثة والحميدة',
    2: 'جراحة عامة',
    3: 'أمراض الجهاز التنفسي',
    4: 'مرض السكري',
    5: 'أمراض الغدد الصماء',
    6: 'ارتفاع ضغط الدم',
    7: 'جراحة العظام',
}

def preprocess_text(text, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=100,  # Adjust according to your model's requirements
        truncation=True,
        padding=True
    )
    return inputs

def predict(text, model, tokenizer):
    model.eval()
    inputs = preprocess_text(text, tokenizer)
    inputs = {k: v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(predictions, dim=-1).item()
        predicted_class = id2label[predicted_class_idx]
        predictions = predictions.cpu().detach()
        prediction_scores = predictions[0].tolist()
    label_scores = {id2label[idx]: score for idx, score in enumerate(prediction_scores)}

    return predicted_class, label_scores

def update_label():
    entered_text = text_entry.get()
    predicted_classes = ["أمراض الدم", "الأورام الخبيثة والحميدة", "جراحة عامة", "أمراض الجهاز التنفسي", "مرض السكري", "أمراض الغدد الصماء", "ارتفاع ضغط الدم", "جراحة العظام"]
    predicted_class, prediction_scores = predict(entered_text, model, tokenizer)
    for row in result_tree.get_children():
        result_tree.delete(row)
    max_score = max(prediction_scores.values())
    for cls in predicted_classes:
        score = prediction_scores.get(cls, 0.0)
        result_tree.insert("", "end", values=(cls, f"{score:.4f}"))
        if score == max_score:
            result_tree.tag_configure('max_score', foreground=highlight_color, font=('Helvetica', 10, 'bold'))
            result_tree.item(result_tree.get_children()[-1], tags=('max_score',))

def ask_for_model_path():
 while True:
        model_path = simpledialog.askstring("Input", "Please enter the path of the BERT model (.pt file):")
        if model_path is None: 
            exit(0)
            return
        elif os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    state_dict = torch.load(f, map_location=torch.device('cpu'))
                    model.load_state_dict(state_dict)
                    messagebox.showinfo("Success", "Model loaded successfully!")
                    return
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {e}")
        else:
            messagebox.showerror("Error", "File not found. Please enter a valid path.")

# Create the main window
root = tk.Tk()
root.title("Arabic Medical QA Classification")

ask_for_model_path()

# Define colors
bg_color = '#f0f0f0'
button_color = '#4CAF50'
text_color = '#333333'
label_color = '#1E90FF'
highlight_color = 'red'

# Styling
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), background=button_color)
style.configure('TLabel', font=('Helvetica', 14), foreground=text_color, background=bg_color)

# Frame for padding and organizing widgets
main_frame = ttk.Frame(root, padding=20)
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Label and Entry
text_label = ttk.Label(main_frame, text="Enter Text:")
text_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

text_entry = ttk.Entry(main_frame, width=40, font=('Helvetica', 14))
text_entry.grid(row=0, column=1, padx=10, pady=10)

# Button
update_button = ttk.Button(main_frame, text="Classify Button", command=update_label)
update_button.grid(row=1, column=0, columnspan=2, pady=10)

# Result Table
result_label = ttk.Label(main_frame, text="Prediction Results:")
result_label.grid(row=2, column=0, columnspan=2, pady=10)

result_tree = ttk.Treeview(main_frame, columns=("Class Name", "Score"), show="headings")
result_tree.heading("Class Name", text="Class Name")
result_tree.heading("Score", text="Score")
result_tree.column("Class Name", width=200, anchor=tk.CENTER)
result_tree.column("Score", width=100, anchor=tk.CENTER)
result_tree.tag_configure('max_score', foreground=highlight_color, font=('Helvetica', 10, 'bold'))
result_tree.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Run the application
root.mainloop()
