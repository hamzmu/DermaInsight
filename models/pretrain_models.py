
from transformers import AutoImageProcessor, AutoModelForImageClassification
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torch
import requests
import os

#Vision Transformer (ViT)
def skintellegent_acne(path: str, top_n = 1) -> dict:
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

    full_result = {}
    for idx, score in enumerate(predictions[0]):
        label = model.config.id2label[idx]
        full_result[label] = round(score.item(), 4)
        
    topk = torch.topk(predictions[0], top_n)
    indices = topk.indices.tolist()
    scores = topk.values.tolist()
        
    topn_result = {}
    for idx, score in zip(indices, scores):
        label = model.config.id2label[idx]
        topn_result[label] = round(score, 4)

    return topn_result

#Vision Transformer (ViT)
def skin_disease_classifier(path: str, top_n = 1) -> dict:
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

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    result = {}
    for idx, score in enumerate(predictions[0]):
        label = model.config.id2label[idx]
        result[label] = round(score.item(), 4)


    else:
        disease_analysis = {
            "Basal Cell Carcinoma": {
                "reason": "Caused by prolonged exposure to ultraviolet (UV) radiation from sunlight or tanning beds, as well as genetic predisposition.",
                "treatment": "Surgical removal, radiation therapy, or topical treatments.",
                "home_remedy": "Apply aloe vera gel to soothe the skin and use green tea extracts for antioxidant benefits."
            },
            "Darier_s Disease": {
                "reason": "A rare genetic disorder caused by mutations in the ATP2A2 gene, leading to issues with skin cell adhesion.",
                "treatment": "Retinoids, moisturizers, and sun protection.",
                "home_remedy": "Use oatmeal baths to relieve irritation and avoid tight clothing to prevent friction."
            },
            "Epidermolysis Bullosa Pruriginosa": {
                "reason": "A rare genetic disorder causing skin fragility and blistering.",
                "treatment": "Wound care, pain management, and avoiding trauma to the skin.",
                "home_remedy": "Apply coconut oil for soothing and keep the skin hydrated with gentle moisturizers."
            },
            "Hailey-Hailey Disease": {
                "reason": "A genetic disorder caused by mutations in the ATP2C1 gene, leading to improper skin cell cohesion.",
                "treatment": "Topical steroids, antibiotics, and avoiding friction or heat.",
                "home_remedy": "Cool compresses and aloe vera gel to relieve discomfort."
            },
            "Herpes Simplex": {
                "reason": "Caused by the herpes simplex virus (HSV), typically transmitted through direct contact or saliva.",
                "treatment": "Antiviral medications like acyclovir or valacyclovir.",
                "home_remedy": "Apply cold compresses or honey to reduce pain and inflammation."
            },
            "Impetigo": {
                "reason": "A bacterial infection caused by Staphylococcus aureus or Streptococcus pyogenes.",
                "treatment": "Topical or oral antibiotics.",
                "home_remedy": "Clean the affected area with diluted vinegar and apply tea tree oil for antimicrobial effects."
            },
            "Larva Migrans": {
                "reason": "Caused by parasitic hookworms that infect the skin, usually through contaminated soil.",
                "treatment": "Anti-parasitic medications like albendazole or ivermectin.",
                "home_remedy": "Soak the affected area in warm water and keep the skin clean."
            },
            "Leprosy Borderline": {
                "reason": "Caused by the bacterium Mycobacterium leprae, typically spread through prolonged close contact.",
                "treatment": "Multi-drug therapy including rifampin, dapsone, and clofazimine.",
                "home_remedy": "Boost immune health with a balanced diet rich in vitamin C and antioxidants."
            },
            "Leprosy Lepromatous": {
                "reason": "A severe form of leprosy caused by Mycobacterium leprae, associated with immune system dysfunction.",
                "treatment": "Long-term multi-drug therapy.",
                "home_remedy": "Include turmeric in the diet for its anti-inflammatory properties."
            },
            "Leprosy Tuberculoid": {
                "reason": "A milder form of leprosy caused by Mycobacterium leprae, with localized skin lesions.",
                "treatment": "Multi-drug therapy including rifampin and dapsone.",
                "home_remedy": "Maintain proper hygiene and support the immune system with vitamin-rich foods."
            },
            "Lichen Planus": {
                "reason": "Thought to be an autoimmune condition triggered by infections, medications, or stress.",
                "treatment": "Topical steroids, antihistamines, and light therapy.",
                "home_remedy": "Apply aloe vera gel to soothe the skin and use turmeric paste for inflammation."
            },
            "Lupus Erythematosus Chronicus Discoides": {
                "reason": "An autoimmune condition triggered by sunlight exposure and genetic factors.",
                "treatment": "Sun protection, topical steroids, and antimalarial drugs.",
                "home_remedy": "Use calendula cream for soothing and avoid sun exposure."
            },
            "Melanoma": {
                "reason": "Caused by mutations in melanocytes, often due to excessive UV radiation exposure and genetic factors.",
                "treatment": "Surgical excision, immunotherapy, or targeted therapy.",
                "home_remedy": "Apply green tea extracts for antioxidant support and avoid sun exposure."
            },
            "Molluscum Contagiosum": {
                "reason": "A viral infection caused by the molluscum contagiosum virus, spread through skin-to-skin contact or contaminated objects.",
                "treatment": "Cryotherapy, topical treatments, or curettage.",
                "home_remedy": "Apply apple cider vinegar as a natural antiseptic."
            },
            "Mycosis Fungoides": {
                "reason": "A type of cutaneous T-cell lymphoma with unknown exact causes but potentially linked to immune dysfunction.",
                "treatment": "Phototherapy, topical treatments, or systemic medications.",
                "home_remedy": "Use coconut oil for hydration and gentle skin care products."
            },
            "Neurofibromatosis": {
                "reason": "A genetic disorder caused by mutations in the NF1 or NF2 genes, leading to benign tumor growth.",
                "treatment": "Surgical removal of tumors and symptom management.",
                "home_remedy": "Maintain a healthy diet and avoid skin irritation."
            },
            "Papilomatosis Confluentes And Reticulate": {
                "reason": "Often associated with genetic factors or chronic irritation.",
                "treatment": "Symptomatic treatment and monitoring.",
                "home_remedy": "Apply aloe vera gel to soothe irritation."
            },
            "Pediculosis Capitis": {
                "reason": "Caused by infestation with head lice (Pediculus humanus capitis), transmitted through close contact.",
                "treatment": "Topical insecticides or manual removal.",
                "home_remedy": "Use a mixture of coconut oil and tea tree oil to remove lice."
            },
            "Pityriasis Rosea": {
                "reason": "Likely caused by viral infections, though the exact virus is unknown.",
                "treatment": "Symptomatic relief with antihistamines or topical treatments.",
                "home_remedy": "Apply calamine lotion for relief and take lukewarm oatmeal baths."
            },
            "Porokeratosis Actinic": {
                "reason": "Caused by prolonged UV exposure or genetic factors, leading to abnormal keratinization.",
                "treatment": "Cryotherapy, topical treatments, or laser therapy.",
                "home_remedy": "Use sunscreen regularly and apply aloe vera for soothing."
            },
            "Psoriasis": {
                "reason": "An autoimmune condition triggered by stress, infections, or genetic predisposition.",
                "treatment": "Topical steroids, phototherapy, or systemic medications.",
                "home_remedy": "Apply coconut oil for moisture and use oatmeal baths for relief."
            },
            "Tinea Corporis": {
                "reason": "A fungal infection caused by dermatophytes, often transmitted through contact with infected individuals or surfaces.",
                "treatment": "Topical or oral antifungal medications.",
                "home_remedy": "Apply tea tree oil to the affected area and keep the skin dry."
            },
            "Tinea Nigra": {
                "reason": "A rare fungal infection caused by Hortaea werneckii, often contracted in tropical regions.",
                "treatment": "Topical antifungal treatments.",
                "home_remedy": "Use apple cider vinegar for cleansing and antifungal effects."
            },
            "Tungiasis": {
                "reason": "Caused by infestation of the skin by the sand flea (Tunga penetrans), often from walking barefoot.",
                "treatment": "Manual removal of fleas and wound care.",
                "home_remedy": "Apply antiseptic and keep the area clean."
            },
            "actinic keratosis": {
                "reason": "Caused by prolonged sun exposure leading to abnormal skin cell changes.",
                "treatment": "Cryotherapy, topical treatments, or laser therapy.",
                "home_remedy": "Use sunscreen and aloe vera for soothing."
            },
            "dermatofibroma": {
                "reason": "Likely caused by minor skin injuries or insect bites, leading to localized fibroblast proliferation.",
                "treatment": "Observation or surgical removal if necessary.",
                "home_remedy": "Apply turmeric paste for natural anti-inflammatory benefits."
            },
            "nevus": {
                "reason": "Usually congenital or caused by genetic mutations in skin cells (melanocytes).",
                "treatment": "Monitoring or surgical removal if changes are observed.",
                "home_remedy": "Apply coconut oil for hydration."
            },
            "pigmented benign keratosis": {
                "reason": "Often caused by aging and prolonged UV exposure.",
                "treatment": "Observation or cryotherapy for cosmetic reasons.",
                "home_remedy": "Use green tea extracts for antioxidant benefits."
            },
            "seborrheic keratosis": {
                "reason": "Caused by aging and genetic factors, with no known environmental triggers.",
                "treatment": "Cryotherapy, curettage, or observation.",
                "home_remedy": "Apply coconut oil for skin hydration."
            },
            "squamous cell carcinoma": {
                "reason": "Caused by prolonged UV exposure, chemical exposure, or chronic skin irritation.",
                "treatment": "Surgical removal, radiation therapy, or topical treatments.",
                "home_remedy": "Use aloe vera for soothing and apply green tea extracts."
            },
            "vascular lesion": {
                "reason": "Caused by abnormal growth or formation of blood vessels, often due to genetic or developmental factors.",
                "treatment": "Laser treatment or surgical intervention.",
                "home_remedy": "Apply cold compresses and use calendula cream for soothing."
            }
        }
        # count of top n we want to include
        topk = torch.topk(predictions, top_n)

        indices = topk.indices.flatten().tolist()
        scores = topk.values.flatten().tolist()

        topn_result = {}

        for idx, score in zip(indices, scores):
            label = model.config.id2label[idx]
            details = disease_analysis.get(label, {
                "reason": "No information available.",
                "treatment": "No information available.",
                "home_remedy": "No information available."
            })
            topn_result[label] = {
                "confidence": round(score, 4),
                "details": details
            }

        return topn_result


#Vision Transformer (ViT)
def skin_type_classifier(path: str, top_n = 1) -> dict:
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

    full_result = {}
    for idx, score in enumerate(predictions[0]):
        label = model.config.id2label[idx]
        full_result[label] = round(score.item(), 4)
        
    topk = torch.topk(predictions[0], top_n)
    indices = topk.indices.tolist()
    scores = topk.values.tolist()
        
    topn_result = {}
    for idx, score in zip(indices, scores):
        label = model.config.id2label[idx]
        topn_result[label] = round(score, 4)

    return topn_result


def parallel_vit_process(path: str) -> dict:
    result = {}
    # pre: only keep top_n <=3
    result["acne"] = skintellegent_acne(path, 2)
    result["disease"] = skin_disease_classifier(path, 2)
    result["type"] = skin_type_classifier(path, 2)
    return result



testing = True
if testing:
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


    #print(skintellegent_acne(filename, True, 1))
    #print(skin_disease_classifier(filename))
    #print(skin_type_classifier(image_path, True))
    print(parallel_vit_process(filename))