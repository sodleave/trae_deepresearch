import os
import json
import logging
import requests
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# 加载环境变量
load_dotenv()

# 配置日志
def setup_logging():
    """配置日志系统，将详细信息写入 deepresearch.log"""
    # 获取 logger
    logger = logging.getLogger('DeepResearch')
    logger.setLevel(logging.DEBUG)  # 记录所有级别的日志
    
    # 防止重复添加 Handler
    if not logger.handlers:
        # 创建文件处理器
        file_handler = logging.FileHandler('deepresearch.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器到 logger
        logger.addHandler(file_handler)
    
    return logger

# 初始化 logger
logger = setup_logging()

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
        error_msg = "错误: 未配置 TAVILY_API_KEY"
        print(error_msg)
        logger.error(error_msg)
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
        logger.info(f"正在执行搜索查询: {query}")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # 记录搜索结果的详细日志
        logger.debug(f"搜索查询 '{query}' 的原始结果:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return result
    except requests.exceptions.RequestException as e:
        error_msg = f"搜索请求失败 ({query}): {e}"
        print(error_msg)
        logger.error(error_msg)
        return None

def decompose_question(query):
    """
    将用户问题拆解为 3-4 个子搜索词
    """
    if not LLM_API_KEY:
        error_msg = "错误: 未配置 LLM_API_KEY"
        print(error_msg)
        logger.error(error_msg)
        return []

    logger.info(f"开始拆解用户问题: {query}")

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    prompt = f"""
    请将以下用户问题拆解为 3-4 个具体的搜索引擎查询词，以便从不同角度获取更全面的信息。
    
    用户问题: {query}
    
    请直接返回一个 JSON 数组，包含这些查询词，不要包含其他解释。
    例如: ["查询词1", "查询词2", "查询词3"]
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的研究助手，擅长将复杂问题拆解为具体的搜索查询。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        logger.debug(f"LLM 拆解问题原始响应: {content}")
        
        # 尝试解析 JSON
        try:
            data = json.loads(content)
            sub_queries = []
            # 兼容可能返回 {"queries": [...]} 或者直接 [...] 的情况
            if isinstance(data, list):
                sub_queries = data
            elif isinstance(data, dict):
                # 寻找可能是列表的值
                for key, value in data.items():
                    if isinstance(value, list):
                        sub_queries = value
                        break
            
            if not sub_queries:
                logger.warning("未能从 LLM 响应中解析出有效的查询列表")
            else:
                logger.info(f"生成的子查询列表: {sub_queries}")
                return sub_queries
                
            return []
        except json.JSONDecodeError:
            error_msg = f"解析拆解问题 JSON 失败: {content}"
            print(error_msg)
            logger.error(error_msg)
            return [query]
            
    except Exception as e:
        error_msg = f"LLM 拆解问题失败: {e}"
        print(error_msg)
        logger.error(error_msg)
        return [query]

def summarize_with_llm(query, all_search_results):
    """
    使用 LLM 总结搜索结果
    """
    if not LLM_API_KEY:
        error_msg = "错误: 未配置 LLM_API_KEY"
        print(error_msg)
        logger.error(error_msg)
        return None

    logger.info(f"开始生成总结，处理 {len(all_search_results)} 个搜索结果集")

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    # 构建提示词
    context = ""
    source_count = 1
    
    # all_search_results 是一个列表，每个元素是一个 search_results 字典
    for search_results in all_search_results:
        if search_results and "results" in search_results:
            for result in search_results["results"]:
                source_entry = f"来源 {source_count}: {result.get('title', '未知标题')}\nURL: {result.get('url', '未知URL')}\n内容: {result.get('content', '')}\n\n"
                context += source_entry
                source_count += 1
    
    logger.debug(f"构建的上下文内容 (前500字符): {context[:500]}...")
    
    prompt = f"""
    请根据以下多方搜索结果回答用户的问题。
    
    用户问题: {query}
    
    搜索结果:
    {context}
    
    请综合上述信息，给出一个清晰、准确、全面的回答。如果搜索结果中没有相关信息，请直接说明。
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个乐于助人的研究助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content
        logger.info("成功生成总结")
        logger.debug(f"生成的总结内容:\n{summary}")
        return summary
    except Exception as e:
        error_msg = f"LLM 调用失败: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None

def main():
    print("=== 深度版 DeepResearch (输入 'quit' 或 'exit' 退出) ===")
    logger.info("程序启动")
    
    # 检查配置
    if not TAVILY_API_KEY or not LLM_API_KEY:
        print("警告: 环境变量未完全配置，请检查 .env 文件")
        logger.warning("环境变量未完全配置")
    
    while True:
        user_input = input("\n请输入你想研究的问题: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            logger.info("用户退出程序")
            break
        if not user_input:
            continue
            
        logger.info(f"收到用户新问题: {user_input}")
        print(f"\n正在分析问题并拆解: {user_input} ...")
        
        sub_queries = decompose_question(user_input)
        
        # 确保原始问题也在搜索列表中
        search_queries = list(set([user_input] + sub_queries))
        print(f"生成的搜索查询: {search_queries}")
        logger.info(f"最终去重后的搜索查询列表: {search_queries}")
        
        print(f"正在并发搜索 {len(search_queries)} 个查询 ...")
        
        all_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 并发执行搜索
            futures = [executor.submit(search_tavily, q) for q in search_queries]
            for future in futures:
                result = future.result()
                if result:
                    all_results.append(result)
        
        if all_results:
            print("搜索完成，正在生成深度总结...")
            summary = summarize_with_llm(user_input, all_results)
            
            print("\n=== 深度回答 ===")
            print(summary)
            print("================")
        else:
            print("搜索未能获取结果，请重试。")
            logger.warning("所有搜索均未返回结果")

if __name__ == "__main__":
    main()
