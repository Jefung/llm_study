import json
import warnings

from langchain_core.prompts import ChatPromptTemplate
from mem0 import Memory

from ai_first.client_init import init_ai_env, create_lc_ai

# 忽略所有警告
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*chromadb.segment.impl.vector.local_persistent_hns.*")


config = {
    "version": "v1.1",
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 1500,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002"
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0db",
            "path": "mem0db",
        }
    },
    "history_db_path": "history.db",
}

# 初始化OpenAI配置
init_ai_env()
print('######################### 例子1: 记忆 ######################################')


mem0 = Memory.from_config(config)

mem0.add("我喜欢读书", user_id="dreamhead", metadata={"category": "hobbies"})
mem0.add("我喜欢编程", user_id="dreamhead", metadata={"category": "hobbies"})

related_memories = mem0.search(query="dreamhead有哪些爱好？", user_id="dreamhead")
print(' '.join([mem["memory"] for mem in related_memories['results']]))

print('######################### 例子2: 使用 mem0 实现长期记忆 ######################################')
# mem0 配置如上例所示
mem0 = Memory.from_config(config)
llm = create_lc_ai()
prompt = ChatPromptTemplate.from_messages([
    ("system", """"你现在扮演孔子的角色，尽量按照孔子的风格回复，不要出现‘子曰’。
    利用提供的上下文进行个性化回复，并记住用户的偏好和以往的交互行为。
    上下文：{context}"""),
    ("user", "{input}")
])
chain = prompt | llm


def retrieve_context(query: str, user_id: str) -> str:
    memories = mem0.search(query, user_id=user_id)
    return ' '.join([mem["memory"] for mem in memories['results']])


def save_interaction(user_id: str, user_input: str, assistant_response: str):
    interaction = [
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]
    # s = json.dumps(interaction)
    # print(s)
    mem0.add(interaction, user_id=user_id)


def invoke(user_input: str, user_id: str) -> str:
    context = retrieve_context(user_input, user_id)
    print('context:', context)
    response = chain.invoke({
        "context": context,
        "input": user_input
    })

    content = response.content
    # print(content)
    save_interaction(user_id, user_input, content)
    return content


user_id = "dreamhead"

while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break

    response = invoke(user_input, user_id)
    print("llm:> " + response)
