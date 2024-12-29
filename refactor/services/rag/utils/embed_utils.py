from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2Model
from io import BytesIO
from PIL import Image
from enum import Enum
from typing import Any, Union, Tuple, List

# --------------------------------------------
# Enum
# --------------------------------------------
class EmbeddingModel(Enum):
    """
    Enum class for embedding models supported
    """
    BLIP    = "BLIP"
    CLIP    = "CLIP"
    DEFAULT =  CLIP

class EmbeddingModelManager:

    # --------------------------------------------
    # Initialization
    # --------------------------------------------
    def __init__(self, model_type: EmbeddingModel = EmbeddingModel.DEFAULT) -> None:
        self.model_type        = model_type
        self.model             = None
        self.processor         = None
        self.get_img_embedding = None
        self.get_txt_embedding = None
        self.output_hiddens    = None
        self.image_dimension   = None
        self.text_dimension    = None
        self.init_model(model_type)
    
    def init_model(self, model_type: EmbeddingModel) -> None:
        try:
            match model_type:

                case EmbeddingModel.BLIP:
                    self.model             = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True)
                    self.processor         = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                    self.get_img_embedding = lambda x: x.pooler_output
                    self.get_txt_embedding = lambda x: x.hidden_states[-1].mean(dim=1)
                    self.output_hiddens    = True
                    self.image_dimension   = 1408
                    self.text_dimension    = 2560
                
                case EmbeddingModel.CLIP:
                    self.model             = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", load_in_8bit=True)
                    self.processor         = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    self.get_img_embedding = lambda x: x
                    self.get_txt_embedding = lambda x: x[0]
                    self.output_hiddens    = False
                    self.image_dimension   = 512
                    self.text_dimension    = 512
                
                case _:
                    self.model_type = EmbeddingModel.DEFAULT
                    self.init_model(EmbeddingModel.DEFAULT)

        except Exception as e:
            raise Exception(f"Error in initializing embedding model: {e}")
    
    # --------------------------------------------
    # Embedding Functions
    # --------------------------------------------
    def embed_image(self, file_content: Union[bytes, BytesIO]) -> Tuple[List[Any], bytes]:
        try:
            # Ensure `file_content` is a BytesIO object
            raw_image = file_content if isinstance(file_content, BytesIO) else BytesIO(file_content)

            # Open and validate the image
            image = Image.open(raw_image).convert("RGB")
            raw_image_bytes = raw_image.getvalue()

            # Preprocess the image and generate embeddings
            inputs = self.processor(images=image, return_tensors="pt", padding=True, truncation=True)
            embeddings = self.get_image_embedding(self.model.get_image_features(**inputs)).cpu().detach().numpy()

            # Return the flattened embeddings and raw image data
            img_embeddings_list = embeddings.flatten().tolist()
            return img_embeddings_list, raw_image_bytes

        except Exception as e:
            raise Exception(f"Invalid image format: {e}")
    
    def embed_text(self, text: str) -> List[Any]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        embeddings = self.get_text_embedding(self.model.get_text_features(**inputs, output_hidden_states=self.output_hidden_states)).cpu().detach().numpy()
        text_embeddings_list = embeddings.flatten().tolist()
        return text_embeddings_list