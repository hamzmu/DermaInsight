
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests
import os

#Vision Transformer (ViT)
def skintellegent_acne(path: str, distribution=False) -> dict:
    """
    https://huggingface.co/imfarzanansari/skintelligent-acne#severity-levels

    Classifies acne severity levels from a facial image.

    Levels:
    - Level -1: Clear Skin
    - Level 0: Occasional Spots
    - Level 1: Mild Acne
    - Level 2: Moderate Acne
    - Level 3: Severe Acne
    - Level 4: Very Severe Acne

    Args:
        path (str): Path to the input image.

    Returns:
        dict: Predicted acne severity level and confidence score.
    """
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained("imfarzanansari/skintelligent-acne")
    model = AutoModelForImageClassification.from_pretrained("imfarzanansari/skintelligent-acne")

    # Load and preprocess image
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():

        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = predictions.argmax().item()
        confidence = predictions[0][predicted_class].item()

    if distribution:    
        result = {}
        for idx, score in enumerate(predictions[0]):
            label = model.config.id2label[idx]
            result[label] = round(score.item(), 4)

        return result
    else:
        result = {
        "class": model.config.id2label[predicted_class],
        "confidence": round(confidence, 4)
        }
        return result

#Vision Transformer (ViT)
def skin_disease_classifier(path: str, distribution=False) -> dict:
    """
    https://huggingface.co/Jayanth2002/dinov2-base-finetuned-SkinDisease

    Classifies skin diseases based on the provided image.

    Supported Skin Disease Classes:
    - Basal Cell Carcinoma
    - Darier's Disease
    - Epidermolysis Bullosa Pruriginosa
    - Hailey-Hailey Disease
    - Herpes Simplex
    - Impetigo
    - Larva Migrans
    - Leprosy (Borderline, Lepromatous, Tuberculoid)
    - Lichen Planus
    - Lupus Erythematosus Chronicus Discoides
    - Melanoma
    - Molluscum Contagiosum
    - Mycosis Fungoides
    - Neurofibromatosis
    - Papillomatosis Confluentes And Reticulate
    - Pediculosis Capitis
    - Pityriasis Rosea
    - Porokeratosis Actinic
    - Psoriasis
    - Tinea Corporis
    - Tinea Nigra
    - Tungiasis
    - Actinic Keratosis
    - Dermatofibroma
    - Nevus
    - Pigmented Benign Keratosis
    - Seborrheic Keratosis
    - Squamous Cell Carcinoma
    - Vascular Lesion

    Args:
        path (str): Path to the input skin image.

    Returns:
        dict: Predicted skin disease class and confidence score.
    """

    # Load model and processor
    processor = AutoImageProcessor.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")
    model = AutoModelForImageClassification.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")

    # Load and preprocess image
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")

    # Inference (no gradients)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = predictions.argmax().item()
        confidence = predictions[0][predicted_class].item()

    if distribution:    
        result = {}
        for idx, score in enumerate(predictions[0]):
            label = model.config.id2label[idx]
            result[label] = round(score.item(), 4)

        return result
    else:
        result = {
        "class": model.config.id2label[predicted_class],
        "confidence": round(confidence, 4)
        }
        return result

#Vision Transformer (ViT)
def skin_type_classifier(path: str, distribution=False) -> dict:
    """
    https://huggingface.co/dima806/skin_types_image_detection

    Classifies skin type from an input image.

    Skin Types:
    - Dry
    - Oily
    - Normal

    Args:
        path (str): Path to the input skin image.

    Returns:
        dict: Predicted skin type and confidence score.
    """
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained("dima806/skin_types_image_detection")
    model = AutoModelForImageClassification.from_pretrained("dima806/skin_types_image_detection")

    # Load and preprocess image
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = predictions.argmax().item()
        confidence = predictions[0][predicted_class].item()

    if distribution:    
        result = {}
        for idx, score in enumerate(predictions[0]):
            label = model.config.id2label[idx]
            result[label] = round(score.item(), 4)

        return result
    else:
        result = {
        "class": model.config.id2label[predicted_class],
        "confidence": round(confidence, 4)
        }
        return result


def skin_cancer_classifier(path: str, distribution=False) -> dict:
    """
    https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification

    Classifies skin cancer type from an input image.

    Cancer Types:
    - Actinic Keratosis
    - Basal Cell Carcinoma
    - Dermatofibroma
    - Melanocytic Nevi
    - Vascular Lesions
    - Benign Keratosis
    - Melanoma
    - Squamous Cell Carcinoma

    Args:
        path (str): Path to the input skin image.
        distribution (bool): Whether to return confidence scores for all classes.

    Returns:
        dict: Predicted skin cancer type and confidence score, or full class distribution if specified.
    """
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")

    # Load and preprocess image
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = predictions.argmax().item()
        confidence = predictions[0][predicted_class].item()

    if distribution:
        result = {}
        for idx, score in enumerate(predictions[0]):
            label = model.config.id2label[idx]
            result[label] = round(score.item(), 4)
        return result
    else:
        result = {
            "class": model.config.id2label[predicted_class],
            "confidence": round(confidence, 4)
        }
        return result


def parallel_vit_process(path: str):
    print("working")


# Image URL
url = "https://www.dermaamin.com/site/images/clinical-pic/n/neurofibromatosis-von-reckling-hausen-syndrome/neurofibromatosis-von-reckling-hausen-syndrome90.jpg"
filename = "neurofibromatosis-von-reckling-hausen-syndrome90.jpg"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

print("working")
# Download the image only if it doesn't exist
import os
if not os.path.exists(filename):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Image saved as {filename}")
    else:
        print("Failed to download image.")
else:
    print("Image already exists.")

# Load and display image
img = Image.open(filename)
img.show()

#print(skintellegent_acne(image_path, True))
print(skin_disease_classifier(filename))
#print(skin_type_classifier(image_path, True))
#print(skin_cancer_classifier(image_path, True))