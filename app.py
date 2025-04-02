import json
from pathlib import Path

import cv2
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import UploadFile, File, Request
from fastrtc import Stream, get_twilio_turn_credentials
from gradio.utils import get_space
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
import numpy as np
import io
from PIL import Image
import base64
import torch

try:
    from yoloe_text_module import create_model_with_text_prompt, predict_with_model
except (ImportError, ModuleNotFoundError):
    print("module not found, please install the required modules")
    


cur_dir = Path(__file__).parent



model = create_model_with_text_prompt(["person"])

# def detection(image, conf_threshold=0.3):
#     image = cv2.resize(image, (model.input_width, model.input_height))
#     print("conf_threshold", conf_threshold)
#     new_image = model.detect_objects(image, conf_threshold)
#     return cv2.resize(new_image, (500, 500))

def detection(image_np, conf_threshold=0.3):
    res_image = predict_with_model(model, image_np, ["person"])
    return cv2.resize(res_image, (500, 500))

stream = Stream(
    handler=detection,
    modality="video",
    mode="send-receive",
    additional_inputs=[gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3)],
    # rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=2 if get_space() else None,
)

app = FastAPI()

stream.mount(app)


@app.get("/")
async def _():
    # rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = open(cur_dir / "index.html").read()
    # html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


class InputData(BaseModel):
    webrtc_id: str
    conf_threshold: float = Field(ge=0, le=1)


@app.post("/input_hook")
async def _(data: InputData):
    stream.set_input(data.webrtc_id, data.conf_threshold)


@app.post("/process_frame")
async def process_frame(request: Request):
    data = await request.json()
    image_data = data["image"].split(",")[1]
    conf_threshold = data.get("conf_threshold", 0.3)

    image_bytes = io.BytesIO(base64.b64decode(image_data))
    pil_image = Image.open(image_bytes).convert("RGB")
    image_np = np.array(pil_image)[:, :, ::-1]  # Convert to BGR for OpenCV

    result = detection(image_np, conf_threshold)

    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)