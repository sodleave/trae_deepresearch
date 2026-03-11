import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 配置
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

def search_tavily(query):
    """
    使用 Tavily API 进行搜索
    """
    if not TAVILY_API_KEY:
        print("错误: 未配置 TAVILY_API_KEY")
        return None
        
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5,
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"搜索请求失败: {e}")
        return None

def summarize_with_llm(query, search_results):
    """
    使用 LLM 总结搜索结果
    """
    if not LLM_API_KEY:
        print("错误: 未配置 LLM_API_KEY")
        return None

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    # 构建提示词
    context = ""
    if search_results and "results" in search_results:
        for i, result in enumerate(search_results["results"]):
            context += f"来源 {i+1}: {result.get('title', '未知标题')}\n"
            context += f"URL: {result.get('url', '未知URL')}\n"
            context += f"内容: {result.get('content', '')}\n\n"
    
    prompt = f"""
    请根据以下搜索结果回答用户的问题。
    
    用户问题: {query}
    
    搜索结果:
    {context}
    
    请综合上述信息，给出一个清晰、准确的回答。如果搜索结果中没有相关信息，请直接说明。
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个乐于助人的研究助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return None

def main():
    print("=== 简易版 DeepResearch (输入 'quit' 或 'exit' 退出) ===")
    
    # 检查配置
    if not TAVILY_API_KEY or not LLM_API_KEY:
        print("警告: 环境变量未完全配置，请检查 .env 文件")
    
    while True:
        user_input = input("\n请输入你想研究的问题: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input:
            continue
            
        print(f"\n正在搜索: {user_input} ...")
        search_results = search_tavily(user_input)
        
        if search_results:
            print("搜索完成，正在生成总结...")
            summary = summarize_with_llm(user_input, search_results)
            
            print("\n=== 回答 ===")
            print(summary)
            print("============")
        else:
            print("搜索未能获取结果，请重试。")

if __name__ == "__main__":
    main()
