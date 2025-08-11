Writing Assistant 📝

A writing assistant tool that combines fine-tuned BART for rephrasing and summarization with grammar correction using language_tool_python. Designed to help users improve text clarity, conciseness, and grammatical accuracy.
🚀 Features

    Rephrasing: Generates alternative phrasings for improved readability.

    Summarization: Condenses long text into concise summaries.

    Grammar Correction: Detects and fixes grammatical, punctuation, and stylistic errors.

    Custom Fine-Tuning: BART model fine-tuned for higher-quality outputs in writing improvement tasks.

🛠 Tech Stack

    Model: BART (Facebook) — fine-tuned for rephrasing & summarization.

    Grammar Check: language_tool_python — offline/online grammar correction API.

    Frameworks & Libraries:

        transformers (Hugging Face)

        torch (PyTorch backend)

        language_tool_python

📂 Project Structure

writing_assistant/
│── best_model/         # Fine-tuned BART model files
│── outputs/            # Generated summaries/rephrasings
│── runs/               # Training logs
│── cache_dir/          # Model cache
│── main.py             # Entry point script
│── requirements.txt    # Project dependencies
│── README.md           # Project documentation

⚙️ Installation

git clone https://github.com/Yeole123/writingassistant.git
cd writingassistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

▶️ Usage

python main.py

Example inside main.py:

from transformers import BartTokenizer, BartForConditionalGeneration
import language_tool_python

# Load fine-tuned BART
tokenizer = BartTokenizer.from_pretrained("best_model")
model = BartForConditionalGeneration.from_pretrained("best_model")

# Grammar tool
tool = language_tool_python.LanguageTool('en-US')

text = "This is an example text which need some correction and summarization."

# Summarization
inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Grammar correction
matches = tool.check(summary)
corrected_text = language_tool_python.utils.correct(summary, matches)

print("Summary:", corrected_text)

📊 Fine-Tuning Details

    Base Model: facebook/bart-large

    Dataset: Custom dataset with paraphrasing & summarization pairs.

    Training: Optimized for high fluency and coherence in generated text.

    Evaluation Metrics: ROUGE, BLEU, Grammar Error Rate.

📌 Future Improvements

    Add UI for user-friendly interaction.

    Support multi-language grammar correction.

    Add API endpoints for integration with other apps.
