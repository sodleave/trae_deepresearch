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

def search_tavily(query):
    """
    Search using Tavily API.
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
        response = _TAVILY_SESSION.post(url, json=payload, timeout=_TAVILY_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        
        # Log search results
        logger.debug(f"搜索查询 '{query}' 的原始结果:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return result
    except requests.exceptions.RequestException as e:
        error_msg = f"搜索请求失败 ({query}): {e}"
        print(error_msg)
        logger.error(error_msg)
        return None
