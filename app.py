import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import pandas as pd

from engine.pro_dubbing_engine import ProDubbingEngine
from engine.models import DubbingSentence

# Load environment variables
load_dotenv()

# --- Streamlit UI --- 
st.set_page_config(layout="wide", page_title="Pro Dubbing Engine V3")

st.title("🎙️ Pro Dubbing Engine Pro V3 - Multi-Step Workflow")

# Initialize session state variables
if "engine" not in st.session_state:
    st.session_state.engine = None
if "step" not in st.session_state:
    st.session_state.step = 1
if "script_content" not in st.session_state:
    st.session_state.script_content = ""
if "translated_srt_content" not in st.session_state:
    st.session_state.translated_srt_content = ""
if "dubbing_sentences" not in st.session_state:
    st.session_state.dubbing_sentences = []
if "final_audio_path" not in st.session_state:
    st.session_state.final_audio_path = None
if "final_srt_content" not in st.session_state:
    st.session_state.final_srt_content = ""

# --- Configuration Sidebar ---
with st.sidebar:
    st.header("Configuration")
    gemini_api_keys_str = st.text_input("Gemini API Keys (comma-separated)", type="password", value=os.getenv("GEMINI_API_KEYS", ""))
    output_language = st.selectbox("Output Language", ["my", "en", "ja", "ko", "th", "vi"], index=0)
    voice_gender = st.selectbox("Voice Gender", ["Male", "Female"], index=0)
    num_workers = st.slider("Number of Parallel Workers", 1, 10, 5)
    tolerance = st.slider("TTS Duration Tolerance (seconds)", 0.1, 1.0, 0.3, 0.1)
    max_ai_retries = st.slider("Max AI Rewriting Retries", 1, 50, 50) # User requested 50
    max_rpm = st.slider("Gemini API RPM (Requests Per Minute)", 1, 60, 9) # User requested 9
    bitrate = st.selectbox("Audio Bitrate", ["96k", "128k", "192k", "256k"], index=2)

    if st.button("Initialize Engine"):
        if not gemini_api_keys_str:
            st.error("Please provide at least one Gemini API Key.")
        else:
            api_keys = [key.strip() for key in gemini_api_keys_str.split(",") if key.strip()]
            if not api_keys:
                st.error("Please provide valid Gemini API Keys.")
            else:
                st.session_state.engine = ProDubbingEngine(
                    api_keys=api_keys,
                    output_language=output_language,
                    voice_gender=voice_gender,
                    tolerance=tolerance,
                    max_ai_retries=max_ai_retries,
                    max_rpm=max_rpm,
                    bitrate=bitrate
                )
                st.success("Pro Dubbing Engine Initialized!")
                st.session_state.step = 1 # Reset to step 1 on re-initialization

# --- Main Content Area ---
if st.session_state.engine is None:
    st.warning("Please initialize the engine in the sidebar first.")
else:
    # Step 1: Script Input & Translation
    if st.session_state.step == 1:
        st.header("Step 1: Script Input & Translate")
        script_input_method = st.radio("Choose input method:", ("Text Input", "Upload .srt/.txt File"))

        if script_input_method == "Text Input":
            st.session_state.script_content = st.text_area("Paste your script here (SRT or plain text)", height=300, value=st.session_state.script_content)
        else:
            uploaded_file = st.file_uploader("Upload .srt or .txt file", type=["srt", "txt"])
            if uploaded_file is not None:
                st.session_state.script_content = uploaded_file.read().decode("utf-8")
                st.text_area("Uploaded Script Content", value=st.session_state.script_content, height=300, disabled=True)

        if st.button("Translate Script"):
            if st.session_state.script_content:
                with st.spinner("Translating script with Gemini AI..."):
                    try:
                        result = st.session_state.engine.run_sync(
                            st.session_state.engine.translate_script(st.session_state.script_content, num_workers)
                        )
                        st.session_state.translated_srt_content = result["reconstructed_srt_content"]
                        st.success("Translation Complete!")
                        st.session_state.step = 2
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during translation: {e}")
                        st.exception(e)
            else:
                st.warning("Please provide script content to translate.")

    # Step 2: Sentence Grouping & Review
    elif st.session_state.step == 2:
        st.header("Step 2: Sentence Grouping & Review")
        st.subheader("Translated SRT Content (Editable)")
        st.session_state.translated_srt_content = st.text_area("", value=st.session_state.translated_srt_content, height=400, key="editable_translated_srt")

        if st.button("Group Sentences"):
            if st.session_state.translated_srt_content:
                with st.spinner("Grouping sentences..."):
                    try:
                        st.session_state.dubbing_sentences = st.session_state.engine.run_sync(
                            st.session_state.engine.group_sentences(st.session_state.translated_srt_content)
                        )
                        st.success(f"Grouped {len(st.session_state.dubbing_sentences)} sentences.")
                        
                        # Display grouped sentences for review
                        sentence_data = []
                        for sentence in st.session_state.dubbing_sentences:
                            sentence_data.append({
                                "ID": sentence.sentence_id,
                                "Start": f"{sentence.start:.2f}",
                                "End": f"{sentence.end:.2f}",
                                "Duration": f"{sentence.duration:.2f}",
                                "Text": sentence.text
                            })
                        st.dataframe(pd.DataFrame(sentence_data), height=300)

                        st.session_state.step = 3
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during sentence grouping: {e}")
                        st.exception(e)
            else:
                st.warning("Please translate script first.")
        
        if st.button("Back to Step 1"):
            st.session_state.step = 1
            st.rerun()

    # Step 3: TTS Generation & AI Adjustment
    elif st.session_state.step == 3:
        st.header("Step 3: TTS Generation & AI Adjustment")
        if not st.session_state.dubbing_sentences:
            st.warning("Please group sentences in Step 2 first.")
            if st.button("Back to Step 2"):
                st.session_state.step = 2
                st.rerun()
        else:
            st.info(f"Ready to generate audio for {len(st.session_state.dubbing_sentences)} sentences.")
            
            # Placeholder for real-time status updates
            status_placeholders = {sentence.sentence_id: st.empty() for sentence in st.session_state.dubbing_sentences}

            def update_status_callback(sentence_id, message):
                if sentence_id in status_placeholders:
                    status_placeholders[sentence_id].text(f"Sentence {sentence_id}: {message}")

            if st.button("Generate Audio"):
                with st.spinner("Generating TTS and adjusting with AI..."):
                    try:
                        st.session_state.dubbing_sentences = st.session_state.engine.run_sync(
                            st.session_state.engine.generate_tts_and_adjust(
                                st.session_state.dubbing_sentences, num_workers, update_status_callback
                            )
                        )
                        st.success("Audio Generation and AI Adjustment Complete!")
                        st.session_state.step = 4
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during audio generation: {e}")
                        st.exception(e)
            
            if st.button("Back to Step 2"):
                st.session_state.step = 2
                st.rerun()

    # Step 4: Final Merging & Download
    elif st.session_state.step == 4:
        st.header("Step 4: Final Merging & Download")
        if not st.session_state.dubbing_sentences:
            st.warning("Please generate audio in Step 3 first.")
            if st.button("Back to Step 3"):
                st.session_state.step = 3
                st.rerun()
        else:
            if st.button("Merge & Finalize"):
                with st.spinner("Merging audio and generating final SRT..."):
                    try:
                        result = st.session_state.engine.run_sync(
                            st.session_state.engine.merge_and_finalize(st.session_state.dubbing_sentences)
                        )
                        st.session_state.final_audio_path = result["final_audio_path"]
                        st.session_state.final_srt_content = result["final_srt_content"]
                        st.success("Finalization Complete!")
                    except Exception as e:
                        st.error(f"Error during finalization: {e}")
                        st.exception(e)
            
            if st.session_state.final_srt_content:
                st.subheader("Final SRT Content")
                st.text_area("", value=st.session_state.final_srt_content, height=400, disabled=True)
                st.download_button(
                    label="Download Final SRT",
                    data=st.session_state.final_srt_content.encode("utf-8"),
                    file_name="final_dubbed.srt",
                    mime="text/plain"
                )
            
            if st.session_state.final_audio_path and os.path.exists(st.session_state.final_audio_path):
                st.subheader("Final Dubbed Audio")
                st.audio(st.session_state.final_audio_path, format="audio/mp3")
                with open(st.session_state.final_audio_path, "rb") as file:
                    st.download_button(
                        label="Download Final Audio (MP3)",
                        data=file.read(),
                        file_name="final_dubbed_audio.mp3",
                        mime="audio/mpeg"
                    )
            
            if st.button("Start Over"):
                st.session_state.step = 1
                st.session_state.script_content = ""
                st.session_state.translated_srt_content = ""
                st.session_state.dubbing_sentences = []
                st.session_state.final_audio_path = None
                st.session_state.final_srt_content = ""
                st.rerun()
            
            if st.button("Back to Step 3"):
                st.session_state.step = 3
                st.rerun()

# Clean up temporary audio files if any
# This part might need more robust handling in a real deployment
# For now, we assume temp_audio directory is cleared on next run or manually
