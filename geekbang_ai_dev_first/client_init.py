import os

import httpx
from openai import OpenAI, DefaultHttpxClient

inited = False
def init_ai_env():
    global inited
    if inited:
        return
    inited = True
    keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_MODEL"
    ]
    print('获取系统环境变量')
    for key in keys:
        print(key, ' : ', os.environ.get(key))

    if not os.environ.get("OPENAI_API_KEY"):
        raise Exception('OPENAI_API_KEY 必须设置')


def create_lc_ai():
    init_ai_env()
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI()
    return chat

def create_open_ai():
    init_ai_env()
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        base_url= os.environ.get('OPENAI_API_BASE'),
        http_client=DefaultHttpxClient(
            proxy= os.environ.get('HTTP_PROXY'),
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        )
    )
    return client

if __name__ == '__main__':
    import os
    from openai import OpenAI, DefaultHttpxClient
    client =create_open_ai()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "who are you?",
            }
        ],
        model="gpt-4o",
    )
#     print(chat_completion)
    print(chat_completion.choices[0].message.content)

    create_lc_ai().invoke("你是谁")