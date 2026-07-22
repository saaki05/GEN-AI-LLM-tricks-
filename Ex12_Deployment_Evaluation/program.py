"""
Ex. No: 12
DEPLOYMENT AND EVALUATION OF A GENERATIVE AI APPLICATION USING
CLOUD-BASED APIs AND AI FRAMEWORKS

Note: demo.launch() is called with blocking=False and share=False here so the script
can run end-to-end unattended and log the app URL + ROUGE scores. The app itself
(inference function, Interface definition) is identical to the manual.
"""
import time
import gradio as gr
from transformers import pipeline
import evaluate

# ---------- 1. Build and Deploy the App ----------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_text(input_text):
    result = summarizer(input_text, max_length=45, min_length=15, do_sample=False)
    return result[0]["summary_text"]


demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=8, label="Enter text to summarize"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="GenAI Text Summarizer",
    description="A cloud-deployable Generative AI summarization app built with Gradio."
)

app, local_url, share_url = demo.launch(share=False, prevent_thread_lock=True)
print(f"Running on local URL: {local_url}")

# quick sanity check of the deployed inference function
sample_text = ("Generative AI refers to a class of artificial intelligence models capable of "
               "producing new content such as text, images, audio, and video.")
print("Sample summarization via deployed function:", summarize_text(sample_text))

time.sleep(2)
demo.close()

# ---------- 2. Evaluate Generated Output ----------
rouge = evaluate.load("rouge")

generated_summaries = [
    "AI models generate new content such as text and images.",
]
reference_summaries = [
    "Generative AI models are capable of producing new content including text and images.",
]

scores = rouge.compute(predictions=generated_summaries, references=reference_summaries)
print("ROUGE Evaluation Scores:", scores)
