# Download test image.
from PIL import Image
from io import BytesIO
from IPython.display import Image as IPImage, display
from huggingface_hub import from_pretrained_keras
import tensorflow as tf
import requests




def google_derm_cnn(path: str) -> dict:
    """
    Runs inference on the given dermatology image and returns the embedding vector.
    Parameters:
        path (str): Path to the image.
    Returns:
        dict: Embedding vector.
    """
    # Load the image
    img = Image.open(path)
    buf = BytesIO()
    img.convert('RGB').save(buf, 'PNG')
    image_bytes = buf.getvalue()

    # Format input
    input_tensor = tf.train.Example(features=tf.train.Features(
        feature={'image/encoded': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))
        })).SerializeToString()

    # Load the model directly from Hugging Face Hub
    loaded_model = from_pretrained_keras("google/derm-foundation")

    # Call inference
    infer = loaded_model.signatures["serving_default"]
    output = infer(inputs=tf.constant([input_tensor]))

    # Extract the embedding vector
    embedding_vector = output['embedding'].numpy().flatten()

    return {"embedding_vector": embedding_vector}







url = "https://storage.googleapis.com/dx-scin-public-data/dataset/images/3445096909671059178.png"
response = requests.get(url)

with open("3445096909671059178.png", "wb") as f:
    f.write(response.content)

print("Image downloaded successfully!")
