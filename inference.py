import torch
from torchvision import transforms,models
import torch.nn  as nn
import json
import gradio as gr

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("labels.json","r") as f:
    labels=json.load(f)

def find_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None

def predict(inp):
    transform= transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformed_img=transform(inp)
    transformed_img=transformed_img.to(device)
    transformed_img=transformed_img.unsqueeze(0)
    preds=model(transformed_img)
    out=torch.argmax(preds,1).item()
    result_key = find_key_by_value(labels, out)
    return (result_key)



model=torch.load('model/dog_breed_model_best.pth',map_location=torch.device('cpu'))
model=model.to(device)



gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Text(),
            ).launch(share=True)