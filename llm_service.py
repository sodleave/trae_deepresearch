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
    Returns a list of URLs ranked by relevance.
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
    
    # If the number of candidates is less than or equal to the limit, return all candidates sorted by default
    # But to support retry mechanism, we should return all candidates if possible or at least a large enough subset
    target_count = max(limit * 2, 10) # Request more URLs to allow for failures
    
    if len(candidates) <= limit:
        # If very few results, just return them all
        return [c["url"] for c in candidates]

    logger.info(f"开始从 {len(candidates)} 个搜索结果中筛选前 {target_count} 个最佳 URL (目标读取: {limit})")
    
    candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2)
    
    prompt = f"""
    请作为一名专业研究员，根据用户问题从以下搜索结果中筛选出最值得深入阅读的网页，并按相关性排序。
    
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
    请返回最相关的 {target_count} 个 URL，按优先级排序。
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
        return [c["url"] for c in candidates[:target_count]]

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
    请仔细阅读以下网页内容，并提取所有与搜索查询相关的有用信息。
    
    搜索查询: {query}
    
    网页内容:
    {truncated_content}
    
    任务要求：
    1. 提取目标：提取任何能部分回答查询、提供背景知识、数据支持或相关细节的信息。
    2. 即使网页内容不能完整回答查询，只要包含与查询相关的有价值信息，都应提取。
    3. 保持信息的原始准确性，不要过度概括。
    4. 如果网页内容完全不包含与搜索查询相关的任何有价值信息，请仅返回"无相关信息"。
    
    请直接返回提取的关键信息点，以列表形式呈现。
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个高效的信息提取助手，擅长从长文中捕捉相关细节。"},
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

def plan_next_step(original_query, confirmed_info=None):
    """
    Plan the next step by analyzing the original query and confirmed information.
    Determines if the original question is fully answered. If not, identifies the
    most important missing piece of information and formulates a simple question for it.
    """
    if not LLM_API_KEY:
        error_msg = "错误: 未配置 LLM_API_KEY"
        print(error_msg)
        logger.error(error_msg)
        return None

    logger.info(f"开始规划下一步，原始问题: {original_query}")
    
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    confirmed_text = ""
    if confirmed_info:
        confirmed_text = "目前已确认的信息:\n" + "\n".join([f"- {info}" for info in confirmed_info]) + "\n\n"
    else:
        confirmed_text = "目前没有任何已确认的信息。\n\n"

    prompt = f"""
    你是一个智能的研究规划助手。你的任务是分析用户的原始问题以及目前已经收集到的信息，规划下一步的搜索动作。
    
    用户原始问题: {original_query}
    
    {confirmed_text}
    
    请仔细分析，执行以下步骤：
    1. 判断目前的“已确认的信息”是否已经足够完整地回答用户的“原始问题”。
    2. 如果信息已经足够，请直接基于已确认的信息生成对原始问题的最终回答。
       - 注意：最终回答必须**足够精简**，直接给出答案，不要包含任何解释、背景介绍或备注。
       - 注意：请根据“用户原始问题”的语境，推测并使用最可能的语言（例如中文或英文）来生成回答。
    3. 如果信息不足，请分析为了回答原始问题，还**缺失**哪些关键信息。
    4. 从缺失的信息中，挑选出**最优先、最核心**需要确认的**一个**信息点。
    5. 将这个最优先的信息点转化为一个**极其简洁、明确的搜索问题**（next_question）。这个新问题不应该像原始问题那样复杂，而应该聚焦于填补当前最大的知识空白。
    
    请返回一个 JSON 对象，格式如下：
    {{
        "is_fully_answered": boolean, // true 表示已收集足够信息可以回答原始问题，false 表示还需要继续搜索
        "final_answer": string, // 如果 is_fully_answered 为 true，基于已确认信息写出最终回答；否则留空
        "missing_info_analysis": string, // 如果 is_fully_answered 为 false，简要分析还缺失什么信息
        "next_question": string // 如果 is_fully_answered 为 false，生成一个简洁明确的下一步搜索问题
    }}
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个逻辑严密的搜索规划助手，擅长将复杂问题拆解为逐步求证的简单问题。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        logger.debug(f"规划下一步原始响应: {content}")
        
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            logger.error(f"解析规划结果 JSON 失败: {content}")
            return None
            
    except Exception as e:
        error_msg = f"LLM 规划失败: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None

def analyze_search_results(current_search_query, all_search_results):
    """
    Analyze search results to determine if the CURRENT simplified query can be answered.
    Extract explicit confirmed information related to the current query.
    """
    if not LLM_API_KEY:
        error_msg = "错误: 未配置 LLM_API_KEY"
        print(error_msg)
        logger.error(error_msg)
        return None

    logger.info(f"开始分析搜索结果，当前查询: {current_search_query}")
    
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    context = ""
    source_count = 1
    
    # Process search results into context string
    for search_results in all_search_results:
        if search_results and "results" in search_results:
            for result in search_results["results"]:
                title = result.get('title', '未知标题')
                url = result.get('url', '未知URL')
                content = result.get('content', '')
                source_entry = f"来源 {source_count}: {title}\nURL: {url}\n内容: {content}\n\n"
                context += source_entry
                source_count += 1
    
    prompt = f"""
    你是一个严谨的信息验证助手。你的任务是根据提供的搜索结果，判断是否能够回答**当前的具体搜索查询**，并提取确认的信息。
    
    当前搜索查询: {current_search_query}
    
    本次搜索结果:
    {context}
    
    请仔细分析上述信息，执行以下步骤：
    1. 判断搜索结果中是否包含了能够直接回答“当前搜索查询”的明确信息。
    2. 如果可以回答，请将得到的明确信息提取出来，作为“新确认的信息”。信息应该具体、客观，可以直接作为后续分析的上下文。
    3. 如果不能回答，说明本次搜索没有找到相关答案。
    
    请返回一个 JSON 对象，格式如下：
    {{
        "is_answered": boolean, // true 表示当前查询被解答，false 表示未找到答案
        "new_confirmed_info": [string] // 如果 is_answered 为 true，列出提取出的新确认信息点；否则为空列表
    }}
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个严谨的研究助手，擅长分析信息完整性。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        logger.debug(f"分析结果原始响应: {content}")
        
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            logger.error(f"解析分析结果 JSON 失败: {content}")
            return None
            
    except Exception as e:
        error_msg = f"LLM 分析失败: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None

def generate_final_answer(original_query, confirmed_info):
    """
    Forcefully generate a concise final answer based on available information.
    """
    if not LLM_API_KEY:
        error_msg = "错误: 未配置 LLM_API_KEY"
        print(error_msg)
        logger.error(error_msg)
        return "无法生成回答: 未配置 LLM_API_KEY"

    logger.info(f"开始强制生成最终回答，原始问题: {original_query}")
    
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    confirmed_text = ""
    if confirmed_info:
        confirmed_text = "目前已确认的信息:\n" + "\n".join([f"- {info}" for info in confirmed_info]) + "\n\n"
    else:
        confirmed_text = "目前没有任何已确认的信息。\n\n"

    prompt = f"""
    你是一个直截了当的问答助手。你的任务是基于目前收集到的有限信息，强制回答用户的原始问题。
    
    用户原始问题: {original_query}
    
    {confirmed_text}
    
    任务要求：
    1. **必须**回答原始问题，即使信息不完整，也要基于现有信息给出最可能的答案。
    2. 回答必须**极其精简**，直接给出核心结论。
    3. **严禁**包含任何解释、背景介绍、"根据搜索结果"、"可能"、"建议"等废话。
    4. **严禁**添加备注或免责声明。
    5. 请分析“用户原始问题”的语境，推测并使用最可能的语言（例如中文或英文）来生成回答。
    
    请直接返回最终答案字符串。
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个只输出核心答案的助手，拒绝任何废话。"},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content.strip()
        logger.debug(f"强制生成回答: {answer}")
        return answer
    except Exception as e:
        error_msg = f"LLM 生成回答失败: {e}"
        print(error_msg)
        logger.error(error_msg)
        return "生成回答失败，请检查日志。"
