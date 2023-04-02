import streamlit as st
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline

model_name = "google/pegasus-xsum"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
example_text=st.text_area("Input text to summarize","")
if st.button("Summarize"):
  nwords=len(example_text.split(" "))
  # Define summarization pipeline 
  summarizer = pipeline("summarization", model=model_name, tokenizer=pegasus_tokenizer,min_length=int(nwords/10)+20, max_length=int(nwords/5+20), framework="pt")
  summary=summarizer(example_text)[0]['summary_text']
  st.text("Summary:")
  st.markdown(summary)
