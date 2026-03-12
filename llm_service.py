import json
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, logger

def decompose_question(query):
    """
    Decompose the user question into 3-4 sub-search queries.
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
        
        try:
            data = json.loads(content)
            sub_queries = []
            
            if isinstance(data, list):
                sub_queries = data
            elif isinstance(data, dict):
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

def select_relevant_urls(query, search_results, limit=5):
    """
    Use LLM to intelligently select the most relevant URLs for deep reading.
    """
    if not search_results or not LLM_API_KEY:
        return []

    # Preprocess results
    candidates = []
    seen_urls = set()
    
    for result_set in search_results:
        if result_set and "results" in result_set:
            for item in result_set["results"]:
                url = item.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    candidates.append({
                        "id": len(candidates),
                        "title": item.get("title", "无标题"),
                        "url": url,
                        "snippet": item.get("content", "")[:200]
                    })
    
    if not candidates:
        return []
        
    if len(candidates) <= limit:
        return [c["url"] for c in candidates]

    logger.info(f"开始从 {len(candidates)} 个搜索结果中筛选前 {limit} 个最佳 URL")
    
    candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2)
    
    prompt = f"""
    请作为一名专业研究员，根据用户问题从以下搜索结果中筛选出最值得深入阅读的网页。
    
    用户问题: {query}
    
    搜索结果列表:
    {candidates_str}
    
    筛选标准：
    1. 相关性：内容必须直接通过事实回答用户问题。
    2. 权威性：优先选择官方文档、知名媒体或技术博客。
    3. 多样性：如果可能，选择不同来源以获得多角度信息。
    4. 信息量：摘要中包含具体细节的优先。
    
    请返回一个 JSON 对象，格式如下：
    {{
        "selected_urls": ["url1", "url2", ...]
    }}
    请只返回 {limit} 个最相关的 URL。
    """

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个擅长筛选高质量信息的研究助手。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        logger.debug(f"URL 筛选原始响应: {content}")
        
        data = json.loads(content)
        selected_urls = data.get("selected_urls", [])
        
        valid_urls = [url for url in selected_urls if url in seen_urls]
        
        logger.info(f"LLM 筛选出的 URL ({len(valid_urls)}个): {valid_urls}")
        return valid_urls
        
    except Exception as e:
        error_msg = f"LLM 筛选 URL 失败: {e}"
        logger.error(error_msg)
        return [c["url"] for c in candidates[:limit]]

def extract_key_info(query, content):
    """
    Extract key information from webpage content using LLM.
    """
    if not content or not LLM_API_KEY:
        return None
        
    max_len = 10000
    truncated_content = content[:max_len]
    if len(content) > max_len:
        truncated_content += "\n...(内容已截断)..."

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    prompt = f"""
    请从以下网页内容中提取与用户问题相关的关键信息。
    
    用户问题: {query}
    
    网页内容:
    {truncated_content}
    
    请提取 3-5 个关键点，简明扼要。如果内容与问题无关，请返回"无相关信息"。
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个高效的信息提取助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        extracted_info = response.choices[0].message.content
        logger.debug(f"提取的关键信息: {extracted_info}")
        return extracted_info
    except Exception as e:
        error_msg = f"LLM 提取信息失败: {e}"
        logger.error(error_msg)
        return None

def summarize_with_llm(query, all_search_results):
    """
    Summarize search results using LLM.
    """
    if not LLM_API_KEY:
        error_msg = "错误: 未配置 LLM_API_KEY"
        print(error_msg)
        logger.error(error_msg)
        return None

    logger.info(f"开始生成总结，处理 {len(all_search_results)} 个搜索结果集")

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    context = ""
    source_count = 1
    
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
