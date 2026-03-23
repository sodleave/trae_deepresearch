import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import TAVILY_API_KEY, logger

_TAVILY_TIMEOUT = (5, 20)
_TAVILY_SESSION = requests.Session()
_TAVILY_SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=20,
        pool_maxsize=20,
        max_retries=Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"POST"},
            raise_on_status=False
        )
    )
)

def _run_tavily_search(payload):
    response = _TAVILY_SESSION.post("https://api.tavily.com/search", json=payload, timeout=_TAVILY_TIMEOUT)
    response.raise_for_status()
    return response.json()

def _merge_results(primary, secondary, max_keep):
    merged = []
    seen = set()
    for block in [primary, secondary]:
        if not block or "results" not in block:
            continue
        for item in block["results"]:
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(item)
            if len(merged) >= max_keep:
                break
        if len(merged) >= max_keep:
            break
    base = primary if primary else {"query": "", "results": []}
    base["results"] = merged
    return base

def search_tavily(query, search_depth="basic", max_results=8, allow_fallback=True):
    """
    Search using Tavily API.
    """
    if not TAVILY_API_KEY:
        error_msg = "错误: 未配置 TAVILY_API_KEY"
        print(error_msg)
        logger.error(error_msg)
        return None
        
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": search_depth,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": max_results,
    }
    
    try:
        logger.info(f"正在执行搜索查询: {query}")
        result = _run_tavily_search(payload)
        primary_count = len(result.get("results", []))

        if allow_fallback and primary_count < max(3, max_results // 2):
            logger.info(f"搜索召回偏低({primary_count})，触发补偿检索: {query}")
            fallback_payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": min(max_results + 4, 12),
            }
            try:
                fallback_result = _run_tavily_search(fallback_payload)
                result = _merge_results(result, fallback_result, min(max_results + 4, 12))
            except requests.exceptions.RequestException as e:
                logger.warning(f"补偿检索失败 ({query}): {e}")
        
        # Log search results
        logger.debug(f"搜索查询 '{query}' 的原始结果:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return result
    except requests.exceptions.RequestException as e:
        error_msg = f"搜索请求失败 ({query}): {e}"
        print(error_msg)
        logger.error(error_msg)
        return None
