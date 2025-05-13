import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Marathi-Sign-Language-Detection"  # Replace with actual path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Marathi label mapping
id2label = {
    "0": "अ", "1": "आ", "2": "इ", "3": "ई", "4": "उ", "5": "ऊ",
    "6": "ए", "7": "ऐ", "8": "ओ", "9": "औ", "10": "क", "11": "क्ष",
    "12": "ख", "13": "ग", "14": "घ", "15": "च", "16": "छ", "17": "ज",
    "18": "ज्ञ", "19": "झ", "20": "ट", "21": "ठ", "22": "ड", "23": "ढ",
    "24": "ण", "25": "त", "26": "थ", "27": "द", "28": "ध", "29": "न",
    "30": "प", "31": "फ", "32": "ब", "33": "भ", "34": "म", "35": "य",
    "36": "र", "37": "ल", "38": "ळ", "39": "व", "40": "श", "41": "स", "42": "ह"
}

def classify_marathi_sign(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_marathi_sign,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5, label="Marathi Sign Classification"),
    title="Marathi-Sign-Language-Detection",
    description="Upload an image of a Marathi sign language hand gesture to identify the corresponding character."
)

if __name__ == "__main__":
    iface.launch()
