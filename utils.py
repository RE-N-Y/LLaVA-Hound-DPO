import torch

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