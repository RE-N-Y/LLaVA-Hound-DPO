import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference, decode2frame

video_path = "examples/sample.mp4"
model_path = "ShareGPTVideo/LLaVA-Hound-DPO"
cache_dir = "/data/tir/projects/tir6/general/sakter/cache"
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, model_name=model_name, cache_dir=cache_dir)
inference_model = ModelInference(model=model, tokenizer=tokenizer, processor=processor, context_len=context_len)

question="What is the evident theme in the video?"
response = inference_model.generate(
    question=question,
    modal_path=video_path,
    video_decode_backend='decord',
    temperature=0,
)
print(response)

# prompt format
