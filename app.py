from pydub import AudioSegment
import whisper
from transformers import pipeline
import streamlit as st

model = whisper.load_model('base')
pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Streamlit app layout
st.title("Video Analysis: Speech-to-Text and Sentiment Analysis")

# File upload
#uploaded_file = st.text_input("Upload a video file path",None)
uploaded_file = st.file_uploader("Upload a video file",type=['mp4'])

if uploaded_file is not None:
    # Convert video to audio
    with st.spinner('Extracting audio from video...'):
        audio = AudioSegment.from_file(uploaded_file)
        audio.export('sample1.mp3', format="mp3")

    with st.spinner('Transcribing audio...'):
        result = model.transcribe('sample1.mp3',word_timestamps=True)

    st.subheader("Sentiment Analysis")
    for txt in result['segments']:
        prompt = txt['text']
        timestmp = txt['start']
        resp = pipe(prompt)
        label = resp[0]['label']
        score = resp[0]['score']
        st.write(f"At {timestmp} '{prompt}' sentiment is {label} with a score of "+"{:.4f}.".format(score))

        