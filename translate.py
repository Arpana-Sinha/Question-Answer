import torch
import gradio as gr
from transformers import pipeline
import json

translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")

with open("language.json" , "r") as file:
    lang_data = json.load(file)

language_names = [item["Language"] for item in lang_data]    

def code(lang):
    for i in lang_data:
        if i['Language'].lower() == lang.lower():
            return i['FLORES-200 code'] 
    return None

def translate(text,tgt_lang):
    tgt_code = code(tgt_lang)
    traslated_text = translator(text,src_lang = "eng_Latn",tgt_lang = tgt_code)
    return traslated_text[0]["translation_text"]

gr.close_all()

demo = gr.Interface(fn=translate,
                    inputs = [gr.Textbox(label="Enter text to be translated", lines = 4),
                              gr.Dropdown(label="Select language", choices = language_names)],
                    outputs= [gr.Textbox(label="Your Translated text", lines = 4)],
                    title="MULTI LANGUAGE-TRANSLATE",
                    description="Translate your english message in any language")

demo.launch()