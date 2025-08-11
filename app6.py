import streamlit as st
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import language_tool_python
from transformers import BartForConditionalGeneration, BartTokenizer

# Set page configuration for a wide layout
st.set_page_config(layout="wide")

st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an Action:", ["Text Rephrasing", "Grammar Correction", "Summarization"])

# Load the fine-tuned BART model for paraphrasing (checkpoint-6020-epoch-1)
model_args = Seq2SeqArgs()
model_args.do_sample = True
model_args.eval_batch_size = 32
model_args.max_length = 64
model_args.num_return_sequences = 1  # Default number of outputs for paraphrasing

paraphrase_model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name=r"outputs\best_model",  # Updated path to fine-tuned model
    args=model_args,
    use_cuda=False
)

# Load BART for summarization
summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load language tool for grammar correction
grammar_tool = language_tool_python.LanguageTool('en-US')

# Function for paraphrasing
# Function for paraphrasing
def paraphrase_text(input_text, num_return_sequences=1):
    model_args.num_return_sequences = num_return_sequences  # Update based on user input
    to_predict = [f"paraphrase: {input_text.strip()}"]
    
    # Get predictions
    predictions = paraphrase_model.predict(to_predict)
    
    # Flatten the list and join split text if needed
    if predictions:
        flat_predictions = [' '.join(prediction.split()) for prediction in predictions]
        return clean_paraphrase_output(flat_predictions)
    else:
        return ["No output generated."]


# Function to clean paraphrase output
def clean_paraphrase_output(predictions):
    cleaned_output = []
    # List of prefixes to remove (adding more combinations)
    prefixes = [
        'aphrase:', 'aphent:', 'aphdomenas:', 'aphrase-', 
        'aphrase :', 'aphent :', 'aphdomenas :', 
        ' aphrase:', ' aphrase :', ' aphent:', 
        ' aphent :', ' aphdomenas:', ' aphdomenas :', 
        'aphrase:', ' aphrase:', ' aphrase : ', 
        'aphdomenas:', 'aphdomenas : ', 'aphdomenas:', 
        ' aphrase: ', 'aphdomenas:', 'aphrase: ',
        ' aphdomenas :', 'aphras:', 'aphras: ', ' aphras :', 'apharas :', 'for ', 'for: '
    ]
    
    for outer_list in predictions:  # Iterate over the outer list
        for p in outer_list:  # Iterate over each inner list
            cleaned_p = p.strip()  # Start by stripping whitespace
            for prefix in prefixes:  # Remove each prefix
                cleaned_p = cleaned_p.replace(prefix, '')
            cleaned_output.append(cleaned_p.strip())  # Add cleaned paraphrase to output list
    return cleaned_output

# Function for grammar correction
def correct_grammar(input_text):
    matches = grammar_tool.check(input_text)
    corrected_text = language_tool_python.utils.correct(input_text, matches)
    return corrected_text

# Function for summarization
def summarize_text(input_text, max_length=150, min_length=30, length_penalty=2.0):
    inputs = summarization_tokenizer.encode(input_text.strip(), return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=6,
        early_stopping=True
    )
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Title and instructions
st.markdown("<h1 style='text-align:center;'>Writing Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Select an option and input your text below.</p>", unsafe_allow_html=True)

# Input area for text
input_text = st.text_area("Input Text", value="Type your text here...", height=200, help="Enter the text you want to process.")

# Additional options for paraphrasing
if option == "Text Rephrasing":
    num_paraphrases = st.sidebar.slider("Number of Paraphrased Outputs", min_value=1, max_value=5, value=1)

# Additional options for summarization
elif option == "Summarization":
    max_length = st.sidebar.slider("Maximum Length of Summary", min_value=50, max_value=300, value=150)
    min_length = st.sidebar.slider("Minimum Length of Summary", min_value=10, max_value=100, value=30)
    length_penalty = st.sidebar.slider("Length Penalty (Higher values mean shorter summaries)", min_value=0.5, max_value=3.0, step=0.1, value=2.0)

# Submit button for processing the text
if st.button("Submit"):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<h3>Input:</h3>", unsafe_allow_html=True)
        st.write(input_text)

    with col2:
        if option == "Text Rephrasing":
            st.markdown("<h3>Rephrased Text:</h3>", unsafe_allow_html=True)
            result = paraphrase_text(input_text, num_return_sequences=num_paraphrases)
            for idx, paraphrase in enumerate(result):
                st.write(f"Paraphrase {idx + 1}: {paraphrase}")
                st.button("Copy Text", key=f"copy_{idx}", on_click=lambda p=paraphrase: st.session_state.setdefault('copy_text', p))

        elif option == "Grammar Correction":
            st.markdown("<h3>Corrected Text:</h3>", unsafe_allow_html=True)
            result = correct_grammar(input_text)
            st.write(result)
            st.button("Copy Text", key="copy_corrected", on_click=lambda: st.session_state.setdefault('copy_text', result))

        elif option == "Summarization":
            st.markdown("<h3>Summary:</h3>", unsafe_allow_html=True)
            result = summarize_text(input_text, max_length=max_length, min_length=min_length, length_penalty=length_penalty)
            st.write(result)
            st.button("Copy Text", key="copy_summary", on_click=lambda: st.session_state.setdefault('copy_text', result))

# Footer
st.markdown("<footer style='text-align:center;'>Powered by BART and LanguageTool</footer>", unsafe_allow_html=True)
