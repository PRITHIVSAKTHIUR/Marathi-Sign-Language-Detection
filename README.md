
![3.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/LLJF46Rwk7iRR_AOXA5vl.png)

# Marathi-Sign-Language-Detection

> Marathi-Sign-Language-Detection is a vision-language model fine-tuned from google/siglip2-base-patch16-224 for multi-class image classification. It is trained to recognize Marathi sign language hand gestures and map them to corresponding Devanagari characters using the SiglipForImageClassification architecture.

```py
Classification Report:
              precision    recall  f1-score   support

           अ     0.9881    0.9911    0.9896      1009
           आ     0.9926    0.9237    0.9569      1022
           इ     0.8132    0.9609    0.8809      1101
           ई     0.9424    0.8894    0.9151      1103
           उ     0.9477    0.9073    0.9271      1198
           ऊ     0.9436    1.0000    0.9710      1071
           ए     0.9153    0.9378    0.9264      1141
           ऐ     0.7790    0.8871    0.8295      1089
           ओ     0.9188    0.9581    0.9381      1075
           औ     1.0000    0.9226    0.9598      1021
           क     0.9566    0.9160    0.9358      1083
           क्ष     0.9287    0.9667    0.9473      1200
           ख     0.9913    1.0000    0.9956      1140
           ग     0.9753    0.9982    0.9866      1109
           घ     0.8398    0.7908    0.8146      1200
           च     0.9388    0.9016    0.9198      1158
           छ     0.9764    0.8127    0.8870      1169
           ज     0.9599    0.9967    0.9779      1200
           ज्ञ     0.9878    0.9483    0.9677      1200
           झ     0.9939    0.9567    0.9749      1200
           ट     0.8917    0.8992    0.8954      1200
           ठ     0.9075    0.8425    0.8738      1200
           ड     0.9354    0.9900    0.9619      1200
           ढ     0.8616    0.9025    0.8816      1200
           ण     0.9114    0.9425    0.9267      1200
           त     0.9280    0.9025    0.9151      1200
           थ     0.9388    0.9717    0.9550      1200
           द     0.8648    0.9275    0.8951      1200
           ध     0.9876    0.9917    0.9896      1200
           न     0.7256    0.8967    0.8021      1200
           प     0.9991    0.9683    0.9835      1200
           फ     0.8909    0.8575    0.8739      1200
           ब     0.9814    0.7917    0.8764      1200
           भ     0.9758    0.8383    0.9018      1200
           म     0.8121    0.8142    0.8132      1200
           य     0.5726    0.9133    0.7039      1200
           र     0.7635    0.7339    0.7484      1210
           ल     0.9239    0.8800    0.9014      1200
           ळ     0.8950    0.7533    0.8181      1200
           व     0.9597    0.7542    0.8446      1200
           श     0.8829    0.8667    0.8747      1200
           स     0.8449    0.8758    0.8601      1200
           ह     0.9604    0.8883    0.9229      1200

    accuracy                         0.9027     50099
   macro avg     0.9117    0.9039    0.9051     50099
weighted avg     0.9107    0.9027    0.9040     50099
```

---

## Label Space: 43 Classes

The model classifies a hand sign into one of the following 43 Marathi characters:

```json
"id2label": {
  "0": "अ", "1": "आ", "2": "इ", "3": "ई", "4": "उ", "5": "ऊ",
  "6": "ए", "7": "ऐ", "8": "ओ", "9": "औ", "10": "क", "11": "क्ष",
  "12": "ख", "13": "ग", "14": "घ", "15": "च", "16": "छ", "17": "ज",
  "18": "ज्ञ", "19": "झ", "20": "ट", "21": "ठ", "22": "ड", "23": "ढ",
  "24": "ण", "25": "त", "26": "थ", "27": "द", "28": "ध", "29": "न",
  "30": "प", "31": "फ", "32": "ब", "33": "भ", "34": "म", "35": "य",
  "36": "र", "37": "ल", "38": "ळ", "39": "व", "40": "श", "41": "स", "42": "ह"
}
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio
```

---

## Inference Code

```python
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
```

---

## Intended Use

Marathi-Sign-Language-Detection can be applied in:

* Educational platforms for learning regional sign language.
* Assistive communication tools for Marathi-speaking users with hearing impairments.
* Interactive applications that translate signs into text.
* Research and data collection for sign language development and recognition.
