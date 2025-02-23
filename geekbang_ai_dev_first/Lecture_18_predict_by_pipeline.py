from threading import Thread

import torch.cuda
from transformers import pipeline

print('############################# 例子：Pipeline使用-文本生成 ###################################')
device = "cuda" if torch.cuda.is_available() else "cpu"

messages = [
    {"role": "user", "content": "请写一首赞美秋天的五言绝句"},
]
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=device, max_new_tokens=100)
result = pipe(messages)
print(result[-1]['generated_text'][-1]['content'])

print('############################# 例子：Pipeline使用-文本翻译 ###################################')
import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device=device)
result = pipe("今天天气真好，我想出去玩。")
print(result[-1]['translation_text'])

print('############################# 例子：Pipeline原始实现 ###################################')

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "user", "content": "请写一首赞美春天的诗，要求不包含春字"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

print('############################# 例子：Pipeline原始实现-流式生成 ###################################')

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "user", "content": "请写一首赞美秋天的五言绝句"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    streamer=streamer,
)

print('############################# 例子：Pipeline-异步流式生成 ###################################')

import torch
from transformers import pipeline, TextIteratorStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

messages = [
    {"role": "user", "content": "请写一首赞美秋天的五言绝句"},
]

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=device, max_new_tokens=100)

streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_kwargs = dict(text_inputs=messages, streamer=streamer)
thread = Thread(target=pipe, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text, end='')