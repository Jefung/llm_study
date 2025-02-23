# [极客时间 - 程序员的 AI 开发第一课](https://time.geekbang.org/column/intro/100839101) 代码示例

Why? 作为技术人员, 最近通过[极客时间 - 程序员的 AI 开发第一课](https://time.geekbang.org/column/intro/100839101)
来入门大模型, 个人习惯通过代码来驱动学习,详细了解落地过程, 其学习过程中将课程代码进行整合并本地跑通.
注: 代码均来源课程

## Python依赖
* Python: `3.10`
* 依赖: `pip install -r requirements.txt`

## 环境变量依赖
> 敏感信息,手动通过系统设置,代码里会使用`os.environ.get获取`
* 必选
  * OPENAI_API_KEY  : `xxxx`
  * OPENAI_MODEL  :  `gpt-3.5-turbo` # 按需填写
* 可选
  * 代理(笔者使用了Clash代理)
      * HTTP_PROXY  :  http://127.0.0.1:7890
      * HTTPS_PROXY  :  http://127.0.0.1:7890
  * OpenAI代理(笔者使用了 [GPT-API-free代理](https://github.com/chatanywhere/GPT_API_free?tab=readme-ov-file)):
    * OPENAI_API_BASE  :  https://api.chatanywhere.tech/v1


## 代码说明
1. OpenAI和LangchainAI的client初始化都是放在client_init.py里面
2. 对应的课程代码在 `Lecture_[课程序号]_[课程名]`
3. 代码细节直接阅读, 按需修改, 可能会包含个人相关注释代码