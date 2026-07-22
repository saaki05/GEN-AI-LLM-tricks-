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
| 4 | [Text Summarization & QA](Ex04_SummarizationQA) | facebook/bart-large-cnn, distilbert-base-cased-distilled-squad | Abstractive summary + extractive QA answer, confidence 0.892 |
| 5 | [Sentiment Analysis & Document Classification](Ex05_SentimentClassification) | distilbert-sst-2, facebook/bart-large-mnli | Sentiment labels (POSITIVE/NEGATIVE, ~1.0 confidence) + zero-shot category probabilities (Economy: 0.68) |
| 6 | [Retrieval-Augmented Generation (RAG)](Ex06_RAG_VectorDB) | all-MiniLM-L6-v2 + FAISS + google/flan-t5-base | Retrieved context chunks + grounded generated answer |
| 7 | [AI-Powered Code Generation & Debugging](Ex07_CodeGenDebug) | Salesforce/codegen-350M-mono | Correctly generated `is_prime()` function + correctly fixed buggy `factorial()` |
| 8 | [Image Generation Using Diffusion Models](Ex08_ImageGeneration_Diffusion) | runwayml/stable-diffusion-v1-5 | `generated_city.png` — 512x512 image synthesised in ~8s (30 steps, fp16, 4GB VRAM) |
| 9 | [Multimodal AI (Text + Image)](Ex09_Multimodal_BLIP) | Salesforce/blip-image-captioning-base, blip-vqa-base | Real caption of the fetched image + VQA answer (see note in folder: the live Unsplash photo is not a dog, unlike the manual's assumption) |
| 10 | [Fine-Tuning DistilBERT](Ex10_FineTuning) | distilbert-base-uncased on IMDB (2000 train / 500 test) | Real 2-epoch fine-tuning run: 61s train time, final eval_accuracy 0.804 |
| 11 | [AI-Based Multimedia Content Generation](Ex11_MultimediaContentGen) | flan-t5-base + stable-diffusion-v1-5 + gTTS | Generated article text, matching image, and narrated audio (mp3) |
| 12 | [Deployment & Evaluation via Gradio](Ex12_Deployment_Evaluation) | facebook/bart-large-cnn + Gradio + ROUGE | Deployed summarizer app (local URL) + ROUGE scores: rouge1 0.61, rouge2 0.38, rougeL 0.61 |

## Real compatibility issues found and fixed while running

Running the manual's code against current (mid-2026) library versions surfaced several genuine
breaking changes, all fixed in the committed scripts/environment rather than papered over:

- **`transformers>=4.50` text-generation pipelines**: a default `max_new_tokens=256` silently
  overrides the manual's `max_length` parameter. Fixed by explicitly passing `max_new_tokens=None`
  (Ex01, Ex02).
- **`transformers 5.x` removed the `summarization`, `question-answering`, and
  `text2text-generation` pipeline task aliases** used throughout the manual. Pinned to
  `transformers<5` (resolved to 4.57.6), and reconciled the resulting `huggingface-hub` version
  requirement with Gradio (which needed `gradio==4.44.1` to match).
- **Gradio 4.44.1 + a too-new Starlette (1.x)**: `demo.launch()`'s internal readiness check threw
  `TypeError: unhashable type: 'dict'` from Jinja2 template caching. Fixed by pinning
  `starlette<1`, `fastapi<1` (resolved to starlette 0.52, fastapi 0.139).

See `requirements.txt` for the exact resolved dependency set used for every run.

## Environment

- Python 3.13, PyTorch 2.6 (CUDA 12.4), Hugging Face `transformers`/`diffusers`/`datasets`
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM) — fp16 + attention slicing used for Stable Diffusion
- All models are pre-trained/public checkpoints pulled from Hugging Face Hub at run time
