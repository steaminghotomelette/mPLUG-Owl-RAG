import torch
from PIL import Image
from decord import VideoReader, cpu
from modelscope import AutoModel, AutoTokenizer
from fastapi import UploadFile
from io import BytesIO
from typing import List, Dict, Union, Any

class MplugOwl3ModelManager:

    # --------------------------------------------
    # Constants
    # --------------------------------------------
    MODEL_NAME = "mPLUG-Owl3"
    MAX_NUM_FRAMES = 128
    MAX_NEW_TOKENS = 500
    MIN_NEW_TOKENS = 0

    # --------------------------------------------
    # Initialization
    # --------------------------------------------
    def __init__(self, model_path: str, attn_implementation: str = "flash_attention_2", device: str = "cuda") -> None:

        # Argument checks
        assert device in ["cuda", "mps"], "Device must be 'cuda' or 'mps'!"
        assert attn_implementation in ["sdpa", "flash_attention_2"], "Attention implementation must be 'sdpa' or 'flash_attention_2'!"

        # Attributes
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None

        # Load the model
        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model = self.model.eval().to(device)

        # Load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = self.model.init_processor(self.tokenizer)
    
    # --------------------------------------------
    # Media Encoding Functions
    # --------------------------------------------
    @classmethod
    def encode_video(cls, video: Union[str, UploadFile]) -> List[Image.Image]:

        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        # Check if path or file like object
        if isinstance(video, str):
            vr = VideoReader(video, ctx=cpu(0))
        else:
            file_bytes = BytesIO()
            file_bytes.write(video.file.read())
            file_bytes.seek(0)
            vr = VideoReader(file_bytes, ctx=cpu(0))
            video.file.seek(0) # Reset file pointer

        sample_fps = round(vr.get_avg_fps() / 1)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]

        if len(frame_idx) > cls.MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, cls.MAX_NUM_FRAMES)

        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        return frames
    
    @classmethod
    def encode_image(cls, image: Union[str, Image.Image, UploadFile]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            file_bytes = BytesIO()
            file_bytes.write(image.file.read())
            file_bytes.seek(0)
            image.file.seek(0) # Reset file pointer
            return Image.open(file_bytes).convert("RGB")
    
    # --------------------------------------------
    # Instance Methods
    # --------------------------------------------
    def respond(
            self,
            messages: List[Dict[str, str]],
            images: Union[List[str], List[UploadFile]] = [],
            videos: Union[List[str], List[UploadFile]] = [],
            gen_params: Dict[str, Any] = [],
            sampling: bool = True,
            streaming: bool = False
    ) -> str:
        
        # Encode input media
        images = [self.encode_image(image) for image in images]
        videos = [self.encode_video(video) for video in videos]

        # Inference
        return self.model.chat(
            images,
            videos,
            messages,
            self.tokenizer,
            processor=self.processor,
            max_new_tokens=self.MAX_NEW_TOKENS,
            min_new_tokens=self.MIN_NEW_TOKENS,
            stream=streaming,
            sampling=sampling,
            **gen_params
        )