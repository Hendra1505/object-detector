import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes


# Custom css 
st.markdown(
    f"""
    <style>
        {open("custom.css").read()}
    </style>
    """,
    unsafe_allow_html=True
)

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta[ "categories"] ## ['_background_", 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train','truck', 'boat', 'traffic']
img_preprocess = weights.transforms() ## Scales values fron 0-255 range to 0-1 range.

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model.eval();## Setting Model for Evaluation/Prediction
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0] ## Dictionary with keys “boxes”, "labels", "scores".
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction): ## Adds Bounding Boxes around original Isage
    img_tensor = torch.tensor(img) ## Transpose
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"], 
                                          colors=["blue" if label=="person" else "red" if label=="car" else "orange" if label=="bicycle" else "green" for label in prediction["labels"]] , width=4)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) ### (3,W,H) -> (W,H,3), Channel first to channel last
    return img_with_bboxes_np

## Tampilan Dashboard
st.title("Object detector menggunakan Algoritma Faster R-CNN")
upload = st.file_uploader(label="Upload Gambar:", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)

    prediction = make_prediction(img) ## Dictionary
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction) ## (U,H,3) -> (3,0,H)
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot (111)
    plt.imshow(img_with_bbox)
    plt.xticks([],[])
    plt.yticks([],[])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]
    st.header("Predicted Probabilities")
    st.write(prediction)