import json
from time import time
from typing import Sequence, Optional, Any

from langchain.globals import set_llm_cache
from langchain_core.caches import InMemoryCache, BaseCache
from langchain_core.outputs import Generation
from langchain_openai import ChatOpenAI

from ai_first.client_init import create_lc_ai

print("########################## 例子:LangChain 中的缓存 ############################")
set_llm_cache(InMemoryCache())

model = create_lc_ai()

start_time = time()
response = model.invoke("给我讲个一句话笑话")
end_time = time()
print(response.content)
print(f"第一次调用耗时: {end_time - start_time}秒")

start_time = time()
response = model.invoke("给我讲个一句话笑话")
end_time = time()
print(response.content)
print(f"第二次调用耗时: {end_time - start_time}秒")


print("########################## 例子: Redis语义缓存(依赖本地启动RedisStack) 为跑通 ############################")
#
# from langchain.globals import set_llm_cache
# from langchain_community.cache import RedisSemanticCache
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#
# RETURN_VAL_TYPE = Sequence[Generation]
#
# def prompt_key(prompt: str) -> str:
#     messages = json.loads(prompt)
#     result = ["('{}', '{}')".format(data['kwargs']['type'], data['kwargs']['content']) for data in messages if
#               'kwargs' in data and 'type' in data['kwargs'] and 'content' in data['kwargs']]
#     return ' '.join(result)
#
#
# class FixedSemanticCache(BaseCache):
#     def __init__(self, cache: BaseCache):
#         self.cache = cache
#
#     def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
#         key = prompt_key(prompt)
#         return self.cache.lookup(key, llm_string)
#
#     def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
#         key = prompt_key(prompt)
#         return self.cache.update(key, llm_string, return_val)
#
#     def clear(self, **kwargs: Any) -> None:
#         return self.cache.clear(**kwargs)
#
# set_llm_cache(
#     FixedSemanticCache(
#         RedisSemanticCache(redis_url="redis://localhost:6379",
#                            embedding=OpenAIEmbeddings())
#     )
# )
#
# model = create_lc_ai()
#
# start_time = time()
# response = model.invoke("""请给我讲一个一句话笑话""")
# end_time = time()
# print(response.content)
# print(f"第一次调用耗时: {end_time - start_time}秒")
#
# start_time = time()
# response = model.invoke("""你能不能给我讲一个一句话笑话""")
# end_time = time()
# print(response.content)
# print(f"第二次调用耗时: {end_time - start_time}秒")