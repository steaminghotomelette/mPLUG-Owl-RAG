import torch
from PIL import Image
from decord import VideoReader, cpu
from modelscope import AutoModel, AutoTokenizer
from typing import List, Dict, Union, Any

class MplugOwl3ModelManager:
    """
    Utility class that exposes mPLUG-Owl3 model operations.
    Provides an interface to interact with the model and handle related media encoding tasks.
    """

    # --------------------------------------------
    # Constants
    # --------------------------------------------
    MODEL_NAME = "mPLUG-Owl3"
    MAX_NUM_FRAMES = 128  # Maximum number of video frames to sample
    MAX_NEW_TOKENS = 2048  # Maximum number of new tokens to generate
    MIN_NEW_TOKENS = 0    # Minimum number of new tokens to generate

    # --------------------------------------------
    # Initialization and Attributes
    # --------------------------------------------
    def __init__(self, model_path: str, attn_implementation: str = "sdpa", device: str = "cuda") -> None:
        """
        Initialize the model manager.

        Args:
            model_path (str): Path to the pretrained model.
            attn_implementation (str): Attention implementation to use ('sdpa' or 'flash_attention_2').
            device (str): Device to load the model on ('cuda' or 'mps').

        Raises:
            AssertionError: If the device or attention implementation is invalid.
            ValueError: If attempting to use int4 model on MPS.
        """
        assert device in ["cuda", "mps"], "Device must be 'cuda' or 'mps'."
        assert attn_implementation in ["sdpa", "flash_attention_2"], "Attention implementation must be 'sdpa' or 'flash_attention_2'."

        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None

        # Load the model
        if "int4" in model_path:
            if device == "mps":
                raise ValueError("Running int4 model with bitsandbytes on Mac is not supported.")
            self.model = AutoModel.from_pretrained(
                model_path, 
                attn_implementation=attn_implementation, 
                trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_path, 
                attn_implementation=attn_implementation, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            )
            self.model = self.model.to(device)

        # Load tokenizer and initialize processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = self.model.init_processor(self.tokenizer)
        self.model.eval()

    # --------------------------------------------
    # Media Encoding Functions
    # --------------------------------------------
    @classmethod
    def encode_video(cls, video: Union[str, object]) -> List[Image.Image]:
        """
        Encode a video into a list of uniformly sampled frames.

        Args:
            video (Union[str, object]): Path to the video file or a file-like object.

        Returns:
            List[Image.Image]: List of sampled video frames as PIL Images.
        """
        def uniform_sample(indices, num_samples):
            """Uniformly sample indices."""
            gap = len(indices) / num_samples
            return [indices[int(i * gap + gap / 2)] for i in range(num_samples)]

        vr = VideoReader(video, ctx=cpu(0))
        frame_indices = list(range(0, len(vr), round(vr.get_avg_fps())))

        # Sample frames if necessary
        if len(frame_indices) > cls.MAX_NUM_FRAMES:
            frame_indices = uniform_sample(frame_indices, cls.MAX_NUM_FRAMES)

        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frame.astype("uint8")) for frame in frames]

    @classmethod
    def encode_image(cls, image: Union[Image.Image, str, object]) -> Image.Image:
        """
        Convert an input to a PIL Image.

        Args:
            image (Union[Image.Image, str, object]): Input image object, file path, or file-like object.

        Returns:
            Image.Image: PIL Image object.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            path = getattr(image, "path", None) or getattr(image.file, "path", None)
            image = Image.open(path).convert("RGB")
        return image

    # --------------------------------------------
    # Chat Handling
    # --------------------------------------------
    def respond(
        self, 
        messages: List[Dict[str, str]], 
        images: Union[List[str], List[object]] = None, 
        videos: Union[List[str], List[object]] = None, 
        streaming: bool = False, 
        params: Dict[str, Any] = None
    ) -> Any:
        """
        Generate a response from the model using provided messages and media inputs.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries containing input texts.
            images (Union[List[str], List[object]], optional): List of image file paths or file-like objects.
            videos (Union[List[str], List[object]], optional): List of video file paths or file-like objects.
            streaming (bool, optional): Whether to use streaming for responses. Defaults to False.
            params (Dict[str, Any], optional): Additional parameters for the model. Defaults to None.

        Returns:
            Any: Model response.
        """
        # Encode media inputs
        encoded_images = [self.encode_image(image) for image in images] if images else []
        encoded_videos = [self.encode_video(video) for video in videos] if videos else []

        # Inference with the model
        return self.model.chat(
            messages=messages,
            images=encoded_images,
            videos=encoded_videos,
            streaming=streaming,
            tokenizer=self.tokenizer,
            processor=self.processor,
            min_new_tokens=self.MIN_NEW_TOKENS,
            max_new_tokens=self.MAX_NEW_TOKENS,
            **(params or {})
        )
