import json
import re
import numpy as np
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, logger

def clean_json_string(json_str):
    """
    Clean JSON string by removing markdown code blocks and other common formatting issues.
    """
    if not isinstance(json_str, str):
        return json_str
        
    # Remove markdown code block syntax
    cleaned = re.sub(r'^```(?:json)?\s*', '', json_str.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # Sometimes LLM might add prefix like "Here is the JSON: "
    # Try to find the first '{' or '[' and last '}' or ']'
    start_idx_dict = cleaned.find('{')
    start_idx_list = cleaned.find('[')
    
    end_idx_dict = cleaned.rfind('}')
    end_idx_list = cleaned.rfind(']')
    
    if start_idx_dict != -1 and end_idx_dict != -1 and (start_idx_list == -1 or start_idx_dict < start_idx_list):
        cleaned = cleaned[start_idx_dict:end_idx_dict + 1]
    elif start_idx_list != -1 and end_idx_list != -1:
        cleaned = cleaned[start_idx_list:end_idx_list + 1]
        
    return cleaned

# 尝试导入 sentence_transformers，如果不存在则在运行时提示
try:
    from sentence_transformers import SentenceTransformer
    # 延迟加载模型，避免在未调用时消耗内存
    _embedding_model = None
    
    def get_embedding_model():
        global _embedding_model
        if _embedding_model is None:
            logger.info("正在加载本地多语言 Embedding 模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
            # 使用支持 50+ 种语言的轻量级多语言模型，适配中英文混合场景
            _embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("本地多语言 Embedding 模型加载完成。")
        return _embedding_model
except ImportError:
    SentenceTransformer = None
    logger.warning("未安装 sentence-transformers，如果需要使用本地过滤，请运行: pip install sentence-transformers")

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
            cleaned_content = clean_json_string(content)
            data = json.loads(cleaned_content)
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
    Use local Embedding model (if available) to efficiently select the most relevant URLs for deep reading.
    Falls back to simple truncation if SentenceTransformer is not installed.
    Returns a list of URLs ranked by relevance.
    """
    if not search_results:
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
    
    target_count = max(limit * 2, 10)
    
    if len(candidates) <= limit:
        return [c["url"] for c in candidates]

    logger.info(f"开始从 {len(candidates)} 个搜索结果中筛选前 {target_count} 个最佳 URL (目标读取: {limit})")
    
    if SentenceTransformer is not None:
        try:
            model = get_embedding_model()
            # 准备待编码文本
            texts = [f"{c['title']} {c['snippet']}" for c in candidates]
            
            # 计算查询和候选文本的 Embedding
            query_embedding = model.encode(query, normalize_embeddings=True)
            doc_embeddings = model.encode(texts, normalize_embeddings=True)
            
            # 计算余弦相似度 (因为已经归一化，点积即为余弦相似度)
            similarities = np.dot(doc_embeddings, query_embedding)
            
            # 排序
            sorted_indices = np.argsort(similarities)[::-1]
            selected_urls = [candidates[i]["url"] for i in sorted_indices[:target_count]]
            
            logger.info(f"本地 Embedding 模型筛选出的 URL ({len(selected_urls)}个): {selected_urls}")
            return selected_urls
        except Exception as e:
            logger.error(f"本地 Embedding 模型筛选失败: {e}，回退到默认排序")
            return [c["url"] for c in candidates[:target_count]]
    else:
        logger.warning("未安装 sentence-transformers，直接返回前N个结果")
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

def plan_next_step(original_query, action_history=None):
    """
    Plan the next step by analyzing the original query and the history of think, act, and observe.
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

    history_text = ""
    if action_history and len(action_history) > 0:
        history_text = "过往探索历史:\n"
        for i, hist in enumerate(action_history):
            history_text += f"第 {i+1} 轮:\n"
            
            think_items = hist.get("think", [])
            if think_items:
                history_text += "  Think (思考):\n"
                for t in think_items:
                    history_text += f"    - {t}\n"
                    
            history_text += f"  Act (动作): {hist.get('act', '未知')}\n"
            
            observe_items = hist.get("observe", [])
            if observe_items:
                history_text += "  Observe (观察结果):\n"
                for o in observe_items:
                    history_text += f"    - {o}\n"
            history_text += "\n"
    else:
        history_text = "目前是第一轮，没有过往探索历史。\n\n"

    prompt = f"""
    你是一个“多跳事实求证”规划器。目标不是泛搜，而是用最少轮次锁定唯一答案。
    你必须基于原始问题与历史观察，规划下一步最有信息增益的动作。

    用户原始问题: {original_query}

    {history_text}

    规划原则（必须遵守）：
    1. 先抽取“硬约束”再搜索：时间范围、地点、身份/头衔、事件、机构、作品、数值区间、语言与输出格式要求。
    2. 优先选择“区分度最高”的锚点作为下一跳，而不是宽泛主题词。
       - 高区分度锚点示例：罕见事件组合、精确年份+机构、独特关系链、限定区间+身份组合。
       - 低区分度锚点示例：泛领域词（如“历史背景”“公司介绍”）。
    3. 每一轮只解决一个关键不确定点，但该点必须是当前瓶颈。
    4. 禁止重复历史上已验证无效或同义的搜索方向；若出现冲突信息，优先规划“判别真伪”的查询。
    5. 当且仅当证据链闭合时才给最终答案：核心实体已唯一、关键约束不冲突、答案格式可直接输出。

    先进行 Think（3-6 条，精炼）：
    - 归纳当前已确认事实（只写关键事实）。
    - 点出尚未闭合的关键缺口。
    - 明确“下一跳为什么是最高优先级”（用区分度/信息增益解释）。
    - 若接近可回答，说明还差的最后一个核验点。

    再进行 Act：
    - 若证据链已闭合：is_fully_answered=true，并输出 final_answer。
      final_answer 必须严格按题目要求格式，极简，不加解释。
    - 若未闭合：is_fully_answered=false，并输出 next_question。
      next_question 必须是“单一、可检索、强约束”的问题句，优先包含可唯一定位的限定词（人名/机构名/年份/地点/作品名/事件名中的至少两个）。

    请返回一个 JSON 对象，格式如下：
    {{
        "think": [string],
        "is_fully_answered": boolean,
        "final_answer": string,
        "next_question": string
    }}
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个严谨的多跳检索规划助手，擅长用高区分度锚点快速缩小搜索空间并完成证据闭环。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        logger.debug(f"规划下一步原始响应: {content}")
        
        try:
            cleaned_content = clean_json_string(content)
            result = json.loads(cleaned_content)
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
    你是一个专业的信息提取助手。你的唯一任务是根据提供的搜索结果，尽可能详尽地提取出与**当前的搜索查询**相关的关键事实和信息。
    
    当前搜索查询: {current_search_query}
    
    本次搜索结果:
    {context}
    
    任务要求：
    1. 仔细阅读所有搜索结果，提取出所有与“当前搜索查询”相关的具体、客观的明确事实信息。
    2. 你不需要判断信息是否完整或是否解答了问题，只要信息有价值且与查询相关，就应该被提取。
    3. 提取的信息应该是独立的、可理解的事实陈述。
    
    请返回一个 JSON 对象，格式如下：
    {{
        "extracted_info": [string] // 列出提取出的所有相关关键信息点；如果没有提取到任何相关信息，则为空列表
    }}
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的研究助手，擅长从长文中提取纯粹的客观事实。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        logger.debug(f"分析结果原始响应: {content}")
        
        try:
            cleaned_content = clean_json_string(content)
            result = json.loads(cleaned_content)
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

def validate_answer(original_query, confirmed_info, candidate_answer):
    """
    Validate if the candidate answer fully resolves the original query based on confirmed info.
    If not, provide the next question to search for.
    """
    if not LLM_API_KEY:
        error_msg = "错误: 未配置 LLM_API_KEY"
        print(error_msg)
        logger.error(error_msg)
        return None

    logger.info(f"开始验证答案: {candidate_answer}")
    
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    confirmed_text = ""
    if confirmed_info:
        confirmed_text = "已确认的信息:\n" + "\n".join([f"- {info}" for info in confirmed_info]) + "\n\n"
    else:
        confirmed_text = "目前没有任何已确认的信息。\n\n"

    prompt = f"""
    你是一个严格的答案验证专家。你的任务是判断给定的“候选答案”是否能够基于“已确认的信息”，完全、准确地回答“原始问题”。
    
    原始问题: {original_query}
    
    {confirmed_text}
    
    候选答案: {candidate_answer}
    
    请仔细分析，执行以下步骤：
    1. 判断候选答案是否直接且完整地回答了原始问题。
    2. 判断候选答案中的事实是否都能在“已确认的信息”中找到依据，不能有捏造。
    3. 如果认为答案合格，"is_correct" 设为 true。
    4. 如果认为答案不合格（如信息不足、未回答核心问题、有事实错误等），"is_correct" 设为 false，并在 "reason" 中说明原因，同时在 "next_question" 中提出下一步需要搜索确认的核心问题。
    
    请返回一个 JSON 对象，格式如下：
    {{
        "is_correct": boolean, // true 表示答案合格，false 表示不合格
        "reason": string, // 如果不合格，简要说明原因；合格则留空
        "next_question": string // 如果不合格，生成一个简洁明确的下一步搜索问题；合格则留空
    }}
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个严谨的验证助手，擅长找出现有答案和信息中的漏洞。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        logger.debug(f"验证答案原始响应: {content}")
        
        try:
            cleaned_content = clean_json_string(content)
            result = json.loads(cleaned_content)
            return result
        except json.JSONDecodeError:
            logger.error(f"解析验证结果 JSON 失败: {content}")
            return None
            
    except Exception as e:
        error_msg = f"LLM 验证失败: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None
