import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from tqdm.auto import tqdm

import json
from dataclasses import dataclass


def tokenize(texts:list[str]|list[tuple[str]], tokenizer, return_tensors="pt", padding="longest"):
    """
    ## Overview
    Tokenize a text (tuple) using the provided tokenizer and returns `input_ids`, `attention_mask`, and `labels`.
    Replace <video> with a special token (-201) in the input_ids and labels.
    
    ## Example (text)
    text = "<s>USER: <video> What is the evident theme in the video? ASSISTANT: The evident theme in the video is that attention is all you need.</s>"
    outputs = tokenize(text, tokenizer)
    {
        "input_ids": tensor([1, ..., -201, ...,  2]),
        "attention_mask": tensor([1, ..., 1, ..., 1]),
        "labels": tensor([1, ..., -201, ...,  2])
    }

    ## Example (tuple)
    text = ("<s>USER: <video> What is the evident theme in the video? ASSITANT:", "The evident theme in the video is that attention is all you need.</s>")
    outputs = tokenize(text, tokenizer)
    {
        "input_ids": tensor([1, ..., -201, ...,  2]),
        "attention_mask": tensor([1, ..., 1, ..., 1]),
        "labels": tensor([-100, ..., -100, XXX, XXX, ..., 2])
    }
    """
    VIDEO = tokenizer.convert_tokens_to_ids("<video>")
    IMAGE = tokenizer.convert_tokens_to_ids("<image>")

    [text, *_] = texts

    if type(text) == str:
        outputs = tokenizer(texts, return_tensors=return_tensors, padding=padding)
        labels = outputs.input_ids.clone()
    elif type(text) == tuple:
        outputs = tokenizer(texts, return_token_type_ids=True, return_tensors=return_tensors, padding=padding)
        labels = outputs.input_ids.clone()
        labels[outputs.token_type_ids == 1] = -100
    else:
        raise ValueError("text should be either a string or a tuple of strings")
    
    outputs.input_ids[outputs.input_ids == VIDEO] = -201
    labels[labels == VIDEO] = -100
    outputs.input_ids[outputs.input_ids == IMAGE] = -200
    labels[labels == IMAGE] = -100
    outputs.labels = labels

    return outputs


@dataclass
class Prompt:
    INSTRUCTION:str
    RESPONSE:str


SFTPROMPT = Prompt(
    INSTRUCTION="<s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <video>\n What do you think the person is going to say next in the video?\n TRANSCRIPT: {context}\n ASSISTANT: ",
    RESPONSE="{response}</s>"
)

class SFTDataset(Dataset):

    def __init__(self, rootdir:Path, prompt:Prompt, processor, tokenizer, completion=True, suffix="merged"):
        """
        rootdir: Path to the root directory of the dataset
        processor: Huggingface processor for tokenizing text captions and video frames

        The dataset should be organised as follows:
        /root
            /<clip>
                /<video>
                /<audio>
                /<metadata.jsonl>
        """
        self.rootdir = rootdir
        self.prompt = prompt
        self.processor = processor
        self.tokenizer = tokenizer
        self.completion = completion
        self.data = []
        
        for clipdir in rootdir.iterdir():
            with open(clipdir / f"{clipdir.name}_{suffix}.jsonl", "r") as f:
                lines = [json.loads(line) for line in f.readlines()]
                for line in lines:
                    self.data.append({
                        "video": {
                            "context": line["src_video"],
                            "response": line["tgt_video"]
                        },
                        "audio": {
                            "context": line["src_audio"],
                            "response": line["tgt_audio"]
                        },
                        "text": {
                            "context": line["src_text"],
                            "response": line["tgt_text"]
                        }
                    })


    def collate_fn(self, samples):
        video_processor = self.processor.get('video')
        video_tensors = video_processor([sample["video"]["context"] for sample in samples], return_tensors="pt", video_decode_backend='decord')
        
        if self.completion:
            INSTURCTION = self.prompt.INSTRUCTION
            RESPONSE = self.prompt.RESPONSE
            prompt = [(INSTURCTION.format(context=sample["text"]["context"]), RESPONSE.format(response=sample["text"]["response"])) for sample in samples]
        else:
            TEMPLATE = self.prompt.INSTRUCTION + self.prompt.RESPONSE
            prompt = [TEMPLATE.format(context=sample["text"]["context"], response=sample["text"]["response"]) for sample in samples]

        outputs = tokenize(prompt, self.tokenizer, padding="longest")

        return {
            "input_ids": outputs.input_ids,
            "attention_mask": outputs.attention_mask,
            "images": [video_tensors["pixel_values"], ["video"]],
            "labels": outputs.labels
        }
    
    def dataloader(self, batch_size=4, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, num_workers=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_path = "ShareGPTVideo/LLaVA-Hound-DPO"
    cache_dir = "/data/tir/projects/tir6/general/sakter/cache"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = None, model_name=model_name, cache_dir=cache_dir)

    path = Path("/data/tir/projects/tir4/data/tvqa_full/friends/raw/data")
    dataset = SFTDataset(path, prompt=SFTPROMPT, processor=processor, tokenizer=tokenizer)
    dataloader = dataset.dataloader()
    sample = next(iter(dataloader))
    model(**sample)