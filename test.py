import google.generativeai as genai
import instructor

from pydantic import BaseModel, Field
from typing import Literal

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
    
    """

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the prediction.",
    )
    label: Literal["Căn cước công dân", "Văn bản pháp luật"] = Field(
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


if __name__ == "__main__":
    for text, label in [
        ("chứng minh nhân dân", "Căn cước công dân"),
        ("văn bản này ban hành năm nào.", "văn bản pháp luật"),
    ]:
        prediction = classify(text)
        #assert prediction.label == label
        print(f"Text: {text}, Predicted Label: {prediction.label}")
        #> Text: Hey Jason! You're awesome, Predicted Label: NOT_SPAM
        #> Text: I am a nigerian prince and I need your help., Predicted Label: SPAM