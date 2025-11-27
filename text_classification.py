import torch
import gradio as gr
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

analyse = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def analyzer(multi_line_text, uploaded_file, column_name):
    texts = []

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file.name)
        except:
            try:
                df = pd.read_csv(uploaded_file.name)
            except Exception as e:
                return f"Failed to read uploaded file: {e}", None

        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available: {list(df.columns)}", None

        texts = df[column_name].astype(str).str.strip().tolist()
        texts = [t for t in texts if t]

    else:
        texts = [line.strip() for line in multi_line_text.splitlines() if line.strip()]

    if not texts:
        return "No valid text found.", None

    results = analyse(texts)
    labels = [r["label"] for r in results]

    labeled_lines = [f"{i+1}. {labels[i]} — {texts[i]}" for i in range(len(texts))]
    labels_text = "\n".join(labeled_lines)

    counts = Counter(labels)
    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", startangle=90)
    ax.axis("equal")

    return labels_text, fig

gr.close_all()

demo = gr.Interface(
    fn=analyzer,
    inputs=[
        gr.Textbox(label="Input text (one per line)", lines=6),
        gr.File(label="Upload Excel/CSV"),
        gr.Textbox(label="Column name for text (required if file uploaded)", placeholder="e.g., review, comments, text")
    ],
    outputs=[
        gr.Textbox(label="Per-line Sentiments", lines=8),
        gr.Plot(label="Sentiment Pie Chart")
    ],
    title="Sentiment Analyzer",
    description="Paste text or upload file, and enter the column name."
)

if __name__ == "__main__":
    demo.launch()
