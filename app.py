import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
import io
from collections import Counter
import numpy as np


# Model Loading

@st.cache_resource(show_spinner=True)
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource(show_spinner=True)
def load_translation_model(src_lang="en", tgt_lang="fr"):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


# Caption Generation

def generate_captions(image, processor, model, max_length=30, num_beams=3, num_return_sequences=1, device='cpu'):
    inputs = processor(images=image, return_tensors="pt").to(device)
    model.to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

def translate_captions(captions, tokenizer, model, device='cpu'):
    model.to(device)
    translated_texts = []
    for caption in captions:
        inputs = tokenizer(caption, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=100)
        tgt_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_texts.append(tgt_text)
    return translated_texts


# Image Preprocessing

def preprocess_image(image, resize_option, custom_size, crop_option):
    if resize_option != "None":
        if resize_option == "Custom":
            if custom_size and len(custom_size) == 2:
                try:
                    w, h = int(custom_size[0]), int(custom_size[1])
                    image = image.resize((w, h))
                except Exception:
                    pass
        else:
            sizes = {
                "256x256": (256, 256),
                "512x512": (512, 512),
                "640x480": (640, 480),
                "800x600": (800, 600),
                "1280×720": (1280, 720),
                "1920×1080": (1920, 1080)
            }
            image = image.resize(sizes.get(resize_option, image.size))

    if crop_option != "None":
        if crop_option == "Center Crop":
            size = min(image.size)
            left = (image.width - size) // 2
            top = (image.height - size) // 2
            image = image.crop((left, top, left + size, top + size))
        elif crop_option == "Square Pad":
            image = ImageOps.pad(image, (max(image.size), max(image.size)))
    return image


# Session Initialization

def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "analytics" not in st.session_state:
        st.session_state.analytics = {
            "total_captions": 0,
            "total_images": 0,
            "language_counter": Counter(),
            "caption_lengths": []
        }


# History & Analytics

def add_to_history(image, captions, translated_captions, translation_lang):
    st.session_state.history.append({
        "image": image,
        "captions": captions,
        "translated_captions": translated_captions,
        "translation_lang": translation_lang
    })
    # Update analytics
    st.session_state.analytics["total_images"] += 1
    st.session_state.analytics["total_captions"] += len(captions)
    if translation_lang != "none":
        st.session_state.analytics["language_counter"][translation_lang] += 1
    for cap in captions:
        st.session_state.analytics["caption_lengths"].append(len(cap.split()))

def display_gallery():
    if not st.session_state.history:
        st.info("No history yet. Upload and caption some images!")
        return
    st.header("🖼️ Captioning History / Gallery")

    cols_per_row = 3
    for idx in range(0, len(st.session_state.history), cols_per_row):
        cols = st.columns(cols_per_row, gap="medium")
        for i, col in enumerate(cols):
            hist_idx = idx + i
            if hist_idx >= len(st.session_state.history):
                break
            entry = st.session_state.history[hist_idx]
            with col:
                st.image(entry["image"], use_container_width=True)
                cap_str = ""
                for j, cap in enumerate(entry["captions"], 1):
                    cap_str += f"Caption {j} (Eng): {cap}\n"
                if entry["translation_lang"] != "none":
                    for j, tcap in enumerate(entry["translated_captions"], 1):
                        cap_str += f"Caption {j} (Trans): {tcap}\n"
                st.text_area("Captions", cap_str.strip(), height=140, key=f"history_captions_{hist_idx}")

                captions_text = ""
                for j, cap in enumerate(entry["captions"], 1):
                    captions_text += f"Caption {j} (English): {cap}\n"
                if entry["translation_lang"] != "none":
                    for j, tcap in enumerate(entry["translated_captions"], 1):
                        captions_text += f"Caption {j} (Translated): {tcap}\n"
                st.download_button(
                    label="Download Captions",
                    data=captions_text,
                    file_name="captions.txt",
                    mime="text/plain",
                    key=f"download_history_{hist_idx}"
                )


# Settings Sidebar

def settings_sidebar():
    st.sidebar.header("⚙️ Caption Generation Settings")

    max_length = st.sidebar.slider(
        "Max Caption Length",
        min_value=5,
        max_value=60,
        value=30,
        step=1,
        help="Maximum number of tokens in the generated caption."
    )
    num_beams = st.sidebar.slider(
        "Beam Search Width",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Controls diversity and quality of captions."
    )
    num_return_sequences = st.sidebar.slider(
        "Number of Captions",
        min_value=1,
        max_value=3,
        value=1,
        step=1,
        help="Generate multiple captions per image."
    )

    lang_options = {
        "No translation (English)": "none",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Italian": "it",
        "Dutch": "nl",
        "Portuguese": "pt",
        "Russian": "ru",
        "Chinese (Simplified)": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Hungarian": "hu",
        "Arabic": "ar",
        "Turkish": "tr"
    }
    translation_choice = st.sidebar.selectbox(
        "Caption Translation",
        options=list(lang_options.keys()),
        help="Translate generated captions to selected language."
    )
    translation_lang = lang_options[translation_choice]

    st.sidebar.header("Image Preprocessing")

    resize_option = st.sidebar.selectbox(
        "Resize Image",
        options=["None", "256x256", "512x512", "640x480", "800x600", "1280×720", "1920×1080", "Custom"],
        help="Resize image before captioning. Choose preset or custom size."
    )
    custom_size = None
    if resize_option == "Custom":
        custom_size_str = st.sidebar.text_input(
            "Custom size (width,height)",
            value="512,512",
            help="Enter width and height separated by comma, e.g. 512,512"
        )
        custom_size = [s.strip() for s in custom_size_str.split(",")]

    crop_option = st.sidebar.selectbox(
        "Crop Image",
        options=["None", "Center Crop", "Square Pad"],
        help="Crop or pad image to improve captioning."
    )

    return max_length, num_beams, num_return_sequences, translation_lang, resize_option, custom_size, crop_option


# Analytics Display

def display_analytics():
    st.header("📊 Analytics Dashboard")
    analytics = st.session_state.analytics

    st.metric("Total Images Processed", analytics["total_images"])
    st.metric("Total Captions Generated", analytics["total_captions"])

    # Most used translation languages (top 5)
    if analytics["language_counter"]:
        most_used = analytics["language_counter"].most_common(5)
        langs_str = ", ".join([f"{lang} ({count})" for lang, count in most_used])
        st.write(f"**Most Used Translation Languages:** {langs_str}")
    else:
        st.write("**Most Used Translation Languages:** None yet")

    # Average caption length
    if analytics["caption_lengths"]:
        avg_len = np.mean(analytics["caption_lengths"])
        st.write(f"**Average Caption Length:** {avg_len:.2f} words")
    else:
        st.write("**Average Caption Length:** No captions generated yet")

    st.markdown("---")
    st.markdown("Analytics data persists only during this session.")


# Main App

def main():
    st.set_page_config(page_title="Image Captioning with BLIP + Analytics", layout="wide")

    # Custom CSS for styling captions & layout
    st.markdown("""
        <style>
            .caption-box {
                background-color: #e0f7fa;
                border-radius: 10px;
                padding: 15px;
                margin: 15px 0;
                font-size: 1.5rem;
                font-weight: 600;
                color: #00796b;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                line-height: 1.4;
            }
            .caption-label {
                font-size: 1.25rem;
                font-weight: 700;
                color: #004d40;
                margin-bottom: 5px;
            }
            .sidebar .sidebar-content {
                padding: 1rem 1.25rem 1rem 1.25rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("AI Image Captioning App with BLIP")
    st.markdown("""
    Upload images, customize preprocessing and captioning parameters, then generate captions with optional translation.  
    Track your usage and stats in the Analytics tab.
    """)

    init_session_state()
    processor, model = load_blip_model()

    (
        max_length,
        num_beams,
        num_return_sequences,
        translation_lang,
        resize_option,
        custom_size,
        crop_option,
    ) = settings_sidebar()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown(f"**Device used:** `{device}`")

    tab1, tab2, tab3 = st.tabs(["Upload & Generate", "History / Gallery", "Analytics"])

    with tab1:
        uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

            processed_image = preprocess_image(image, resize_option, custom_size, crop_option)
            st.image(processed_image, caption="Processed Image Preview", use_container_width=True)

            if st.button("Generate Caption"):
                with st.spinner("Generating caption(s), please wait..."):
                    captions = generate_captions(
                        processed_image,
                        processor,
                        model,
                        max_length=max_length,
                        num_beams=num_beams,
                        num_return_sequences=num_return_sequences,
                        device=device
                    )

                    translated_captions = captions
                    if translation_lang != "none":
                        try:
                            tokenizer, translation_model = load_translation_model("en", translation_lang)
                            translated_captions = translate_captions(captions, tokenizer, translation_model, device=device)
                        except Exception as e:
                            st.error(f"Translation model loading failed: {e}")
                            translated_captions = captions

                    # Show captions prominently
                    for i, cap in enumerate(captions):
                        st.markdown(f'<div class="caption-label">Caption {i+1} (English):</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="caption-box">{cap}</div>', unsafe_allow_html=True)
                        if translation_lang != "none":
                            st.markdown(f'<div class="caption-label">Caption {i+1} (Translated):</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="caption-box">{translated_captions[i]}</div>', unsafe_allow_html=True)

                    # Download button for captions
                    captions_text = ""
                    for i, cap in enumerate(captions):
                        captions_text += f"Caption {i+1} (English): {cap}\n"
                    if translation_lang != "none":
                        for i, tcap in enumerate(translated_captions):
                            captions_text += f"Caption {i+1} (Translated): {tcap}\n"

                    st.download_button(
                        label="Download Caption",
                        data=captions_text,
                        file_name="captions.txt",
                        mime="text/plain"
                    )

                    add_to_history(processed_image, captions, translated_captions, translation_lang)

        else:
            st.info("Please upload an image to generate captions.")

    with tab2:
        display_gallery()

    with tab3:
        display_analytics()


if __name__ == "__main__":
    main()
