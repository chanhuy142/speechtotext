import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import transformers
import google.generativeai as genai
import instructor

import wavio
from pydantic import BaseModel, Field
from typing import Literal
print(sd.query_devices())
genai.configure(api_key="AIzaSyBdFQ0hs5-bnR5B07Co1TgeUEhUONUhO9k")
client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name="models/gemini-1.5-flash-latest",  
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )

class ClassificationResponse(BaseModel):
    """
    A few-shot example of text classification:

    Examples:
    - "Cho tôi xem chứng minh nhân dân cá nhân của tôi với!": Căn cước công dân
    - "Văn bản này ban hành năm nào.": Văn bản pháp luật
    - "Nhà đất này có giấy tờ chưa": Văn bản pháp luật
    - "Cho tôi xem bằng lái xe của tôi với!": Căn cước công dân
    - "Hợp đồng này ký kết vào thời gian nào?": Văn bản pháp luật
    - "Thẻ căn cước của tôi bị mất, tôi có thể xin cấp lại không?": Căn cước công dân
    - "Văn bản này có điều khoản bổ sung nào không?": Văn bản pháp luật
    - "Tôi cần căn cước để mở hợp đồng lao động.": Căn cước công dân
    - "Văn bản này có hiệu lực từ ngày nào?": Văn bản pháp luật
    - "Làm thế nào để xác thực căn cước của tôi?": Căn cước công dân
    - "Nhà này có sổ đỏ không": Văn bản pháp luật
    
    """

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the prediction.",
    )
    label: Literal["Căn cước công dân", "Văn bản pháp luật","Khác"] = Field(
        ...,
        description="The predicted class label.",
    )

def classify(data: str) -> ClassificationResponse:
    """Perform single-label classification on the input text."""
    return client.chat.completions.create(
        
        response_model=ClassificationResponse,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: <text>{data}</text>",
            },
        ],
        
    )



# Cài đặt thông số ghi âm
duration = 2  # Thời gian ghi âm (giây)
sample_rate = 16000  # Tần số lấy mẫu (Hz)
filename = "recording.wav"  # File lưu tạm thời
# Hàm ghi âm từ mic và trả về dữ liệu âm thanh
def record_audio():
    st.write("Đang ghi âm...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32',device=1)
    sd.wait()  # Chờ ghi âm xong
    wavio.write(filename, audio, sample_rate, sampwidth=2)  # Lưu thành file wav
    st.write("Đã ghi âm xong!")
    return audio.flatten()
  # Trả về mảng 1D để đưa vào Whisper

# Hàm chuyển âm thanh thành văn bản bằng Whisper
def transcribe_audio(audio_data):
    model = whisper.load_model("base", device="cuda")  # Tải mô hình Whisper
    audio_data = whisper.pad_or_trim(audio_data)  # Điều chỉnh độ dài dữ liệu âm thanh nếu cần
    mel = whisper.log_mel_spectrogram(audio_data)  # Chuyển đổi thành dạng log-Mel spectrogram
    result = model.transcribe(audio_data, fp16=False, language="vi")  # Chuyển thành văn bản
    return result["text"]



# Giao diện Streamlit
st.title("Ứng dụng Speech-to-Text")

if st.button("Bấm để ghi âm"):
    audio_data = record_audio()  # Ghi âm trực tiếp từ mic
    text = transcribe_audio( audio_data )  # Chuyển âm thanh thành văn bản
    
    
    st.write("Nội dung bạn vừa nói là:")
    st.write(text)
    label = classify(text)  # Phân loại văn bản
    st.write("Phân loại:")
    st.write(label)
    if(label.label == "Căn cước công dân"):
        model=genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction="Bạn là một trợ lí ảo trả lời về căn cước công dân, các câu hỏi được chuyển từ âm thanh sang text nên sẽ có sai sót, hãy điều chỉnh sai sót này trong câu trả lời của bạn")
        response = model.generate_content(text)
        st.write(response.text)
    else:
        model=genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction="Bạn là một trợ lí ảo trả lời về văn bản pháp luật và là 1 chuyên gia pháp lí, các câu hỏi được chuyển từ âm thanh sang text nên sẽ có sai sót, hãy điều chỉnh sai sót này trong câu trả lời của bạn")
        response = model.generate_content(text)
        st.write(response.text)
