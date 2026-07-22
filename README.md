# CS4V48 - GenAI and LLM Laboratory

Implementation, execution and output records for all 12 experiments of the
**CS4V48 GenAI & LLM Laboratory Manual**. Every experiment was actually run on a
local machine (Windows, NVIDIA RTX 3050 Ti, CUDA) using the Hugging Face
`transformers` / `diffusers` ecosystem — this is not just a code listing, each
folder contains the real captured console output, generated artifacts
(images/audio), and a screenshot of the run.

## Repository layout

Each `ExNN_*` folder contains:
- `program.py` — the experiment's source code
- `output.txt` — captured console output from the actual run
- `screenshot.png` — a terminal-style screenshot of the run's output
- any generated artifacts (images, audio) produced by that experiment

## Experiments and outputs

| # | Experiment | Model(s) used | Output |
|---|---|---|---|
| 1 | [Text Generation Using Pre-Trained Foundation Models](Ex01_TextGeneration) | GPT-2 | Two sampled text continuations of a seed prompt |
| 2 | [Prompt Engineering Techniques](Ex02_PromptEngineering) | GPT-2 | Zero-shot, few-shot, and chain-of-thought prompt outputs compared |
| 3 | [Conversational AI Chatbot](Ex03_ConversationalChatbot) | microsoft/DialoGPT-medium | Multi-turn dialogue log with context retained across turns |
| 4 | [Text Summarization & QA](Ex04_SummarizationQA) | facebook/bart-large-cnn, distilbert-base-cased-distilled-squad | Abstractive summary + extractive QA answer with confidence score |
| 5 | [Sentiment Analysis & Document Classification](Ex05_SentimentClassification) | distilbert-sst-2, facebook/bart-large-mnli | Sentiment labels + zero-shot category probabilities |
| 6 | [Retrieval-Augmented Generation (RAG)](Ex06_RAG_VectorDB) | all-MiniLM-L6-v2 + FAISS + google/flan-t5-base | Retrieved context chunks + grounded generated answer |
| 7 | [AI-Powered Code Generation & Debugging](Ex07_CodeGenDebug) | Salesforce/codegen-350M-mono | Generated `is_prime()` function + fixed buggy `factorial()` |
| 8 | [Image Generation Using Diffusion Models](Ex08_ImageGeneration_Diffusion) | runwayml/stable-diffusion-v1-5 | `generated_city.png` synthesised from a text prompt |
| 9 | [Multimodal AI (Text + Image)](Ex09_Multimodal_BLIP) | Salesforce/blip-image-captioning-base, blip-vqa-base | Image caption + visual question answering result |
| 10 | [Fine-Tuning DistilBERT](Ex10_FineTuning) | distilbert-base-uncased on IMDB | Training/eval loss & accuracy after 2-epoch fine-tuning |
| 11 | [AI-Based Multimedia Content Generation](Ex11_MultimediaContentGen) | flan-t5-base + stable-diffusion-v1-5 + gTTS | Generated article text, matching image, and narrated audio (mp3) |
| 12 | [Deployment & Evaluation via Gradio](Ex12_Deployment_Evaluation) | facebook/bart-large-cnn + Gradio + ROUGE | Deployed summarizer app + ROUGE evaluation scores |

## Environment

- Python 3.13, PyTorch 2.6 (CUDA 12.4), Hugging Face `transformers`/`diffusers`/`datasets`
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM) — fp16 + attention slicing used for Stable Diffusion
- All models are pre-trained/public checkpoints pulled from Hugging Face Hub at run time
