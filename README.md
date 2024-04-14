# LLaVA

## Prompt format

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <video>
What is the evident theme in the video? ASSISTANT:
```

## Input IDs 
```
input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['video'], return_tensors='pt')

1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 29871, -201, 29871, 13, 5618, 338, 278, 13602, 10929, 297, 278, 4863, 29973, 319, 1799, 9047, 13566, 2990
```

## Video frames

```
modal_path = "frames/..."
modal_tensor = video_processor(modal_path, return_tensors='pt', video_decode_backend=video_decode_backend)

[
    [tensor(3, 8, 224, 224), tensor(3, 8, 224, 224), ...], 
    ['image',                'image'               , ...]
]
```
