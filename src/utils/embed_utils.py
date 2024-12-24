from fastapi import HTTPException
from transformers import CLIPProcessor, CLIPModel, Blip2Model, AutoProcessor
from io import BytesIO
from PIL import Image
from enum import Enum

# Load models
try:
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", load_in_8bit=True)

    # Temporarily disabled
    # blip_model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True)
    # blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = None
    blip_processor = None
except Exception as e:
    raise Exception(f"Error loading models: {e}")


class EmbeddingModel(Enum):
    """
    Enum class for embedding models supported
    """
    BLIP = "BLIP"
    CLIP = "CLIP"
    DEFAULT = CLIP


class EmbeddingModelManager:
    def __init__(self, model_type: EmbeddingModel = EmbeddingModel.DEFAULT):
        self.model_type = model_type
        self.model = None
        self.processor = None
        self.get_image_embedding = None
        self.get_text_embedding = None
        self.output_hidden_states = True
        self.image_dimension = 0
        self.text_dimension = 0
        self.set_model(model_type)

    def set_model(self, model_type: EmbeddingModel | str):
        """
        Sets the model and processor based on the specified model type.

        Args:
            model_type (EmbeddingModel): The type of embedding model to be used. It determines which model and processor are loaded.

        """
        try:
            if isinstance(model_type, str):
                model_type = EmbeddingModel(model_type)
            match model_type:
                case EmbeddingModel.BLIP:
                    self.model_type = model_type
                    self.model = blip_model
                    self.processor = blip_processor
                    self.get_image_embedding = lambda x: x.pooler_output
                    self.get_text_embedding = lambda x: x.hidden_states[-1].mean(
                        dim=1)
                    self.output_hidden_states = True
                    self.image_dimension = 1408
                    self.text_dimension = 2560

                case EmbeddingModel.CLIP:
                    self.model_type = model_type
                    self.model = clip_model
                    self.processor = clip_processor
                    self.get_image_embedding = lambda x: x
                    self.get_text_embedding = lambda x: x[0]
                    self.output_hidden_states = False
                    self.image_dimension = 512
                    self.text_dimension = 512

                case _:
                    self.model_type = EmbeddingModel.DEFAULT
                    self.set_model(model_type=EmbeddingModel.DEFAULT)
        except Exception as e:
            raise Exception(f"Error in setting model: {model_type}")

    def embed_image(self, file_content):
        """
        Processes an image from raw byte content, generates its embeddings, and returns both the embeddings and raw image data.

        Args:
            file_content (bytes): The raw byte content of the image to be processed.

        Returns:
            tuple: A tuple containing two elements:
                - img_embeddings_list (list): A flattened list of the image's embeddings.
                - raw_image (bytes): The raw byte content of the image after reading and processing.

        """
        try:
            if isinstance(file_content, bytes):
                raw_image = BytesIO(file_content)
            else:
                assert (isinstance(file_content, BytesIO))
                raw_image = file_content
            image = Image.open(raw_image)
            raw_image = raw_image.getvalue()
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image format: {e}")

        inputs = self.processor(images=image, return_tensors="pt",
                                padding=True, truncation=True)
        embeddings = self.get_image_embedding(
            self.model.get_image_features(**inputs)).cpu().detach().numpy()
        img_embeddings_list = embeddings.flatten().tolist()
        return img_embeddings_list, raw_image

    def embed_text(self, text: str):
        """
        Processes a given text and generates its embeddings.

        Args:
            text (str): The input text to be processed and embedded.

        Returns:
            list: List of text embeddings

        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        embeddings = self.get_text_embedding(
            self.model.get_text_features(**inputs,
                                         output_hidden_states=self.output_hidden_states)).cpu().detach().numpy()
        text_embeddings_list = embeddings.flatten().tolist()
        return text_embeddings_list
