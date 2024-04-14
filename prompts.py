from dataclasses import dataclass

@dataclass
class Prompt:
    INSTRUCTION:str
    RESPONSE:str


SFTPROMPT = Prompt(
    INSTRUCTION="<s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <video>\n What do you think the person is going to say next in the video?\n TRANSCRIPT: {context}\n ASSISTANT: ",
    RESPONSE="{response}</s>"
)