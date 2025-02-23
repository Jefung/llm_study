import torch
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

device = 0 if torch.cuda.is_available() else -1

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    device=device,
    pipeline_kwargs=dict(
        max_new_tokens=512,
        return_full_text=False,
    ),
)


chat_model = ChatHuggingFace(llm=llm)


result = chat_model.invoke("写一首赞美秋天的五言绝句。")
print(result.content)