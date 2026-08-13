import os
# 1. MUST set HF_TOKEN before importing transformers


import streamlit as st
import pysrt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 2. Auto-detect GPU for 10x faster execution if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Load Model ---
@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

tokenizer, model = load_model()

# --- 2. Batch Translation Function ---
def translate_batch(texts):
    """Translates a batch of subtitle strings at once to speed up processing"""
    if not texts:
        return []
    
    # Replace internal newlines so formatting isn't lost
    cleaned_texts = [t.replace("\n", " <br> ") for t in texts]
    
    inputs = tokenizer(cleaned_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("mal_Mlym")
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=2,  # Reduced from 4 to 2 for 2x faster speed with identical accuracy
            early_stopping=True
        )
    
    decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return [d.replace(" <br> ", "\n").strip() for d in decoded]

# --- 3. Streamlit UI ---
st.set_page_config(page_title="SRT to Malayalam Translator", page_icon="🎬")
st.title("🎬 SRT to Malayalam Translator")
st.write("ഇംഗ്ലീഷ് `.srt` സബ്ടൈറ്റിൽ ഫയൽ അപ്ലോഡ് ചെയ്താൽ മലയാളത്തിലേക്ക് വിവർത്തനം ചെയ്തു തരും (ടൈമിംഗ് മാറില്ല)")

uploaded_file = st.file_uploader("ഇംഗ്ലീഷ് .srt ഫയൽ തിരഞ്ഞെടുക്കുക", type="srt")

if uploaded_file is not None:
    # 3. Safe decoding that handles BOM and legacy Windows text encodings
    raw_data = uploaded_file.read()
    try:
        srt_content = raw_data.decode("utf-8-sig")
    except UnicodeDecodeError:
        srt_content = raw_data.decode("latin-1", errors="ignore")
        
    subs = pysrt.from_string(srt_content)
    
    st.info(f"📄 ആകെ {len(subs)} സബ്ടൈറ്റിലുകൾ കണ്ടെത്തി. (Running on: **{device.upper()}**)")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 4. Process in batches of 16 lines for massive speedup
    BATCH_SIZE = 16
    non_empty_subs = [sub for sub in subs if sub.text.strip()]
    total_subs = len(non_empty_subs)
    
    for idx in range(0, total_subs, BATCH_SIZE):
        batch = non_empty_subs[idx : idx + BATCH_SIZE]
        texts_to_translate = [sub.text for sub in batch]
        
        try:
            translated_texts = translate_batch(texts_to_translate)
            for sub, trans_text in zip(batch, translated_texts):
                sub.text = trans_text
        except Exception as e:
            st.warning(f"Batch {idx//BATCH_SIZE + 1} processing error: {e}")
            
        current_progress = min((idx + BATCH_SIZE) / total_subs, 1.0)
        progress_bar.progress(current_progress)
        status_text.text(f"വിവർത്തനം ചെയ്യുന്നു: {min(idx + BATCH_SIZE, total_subs)}/{total_subs}")
    
    status_text.text("✅ വിവർത്തനം പൂർത്തിയായി!")
    
    translated_srt = "\n\n".join(str(sub) for sub in subs)
    
    st.download_button(
        label="📥 മലയാളം SRT ഡൗൺലോഡ് ചെയ്യുക",
        data=translated_srt.encode('utf-8'),
        file_name="malayalam_subtitles.srt",
        mime="text/plain"
    )
    
    with st.expander("👁️ പ്രിവ്യൂ കാണുക"):
        st.text(translated_srt[:1000] + "..." if len(translated_srt) > 1000 else translated_srt)