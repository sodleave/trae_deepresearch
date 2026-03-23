import requests
import re
from html import unescape
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import logger

_JINA_TIMEOUT = (5, 20)
_JINA_SESSION = requests.Session()
_JINA_SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=20,
        pool_maxsize=20,
        max_retries=Retry(
            total=2,
            backoff_factor=0.4,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"GET"},
            raise_on_status=False
        )
    )
)

_DIRECT_TIMEOUT = (5, 12)
_DIRECT_SESSION = requests.Session()
_DIRECT_SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=20,
        pool_maxsize=20,
        max_retries=Retry(
            total=1,
            backoff_factor=0.2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"GET"},
            raise_on_status=False
        )
    )
)

def _extract_readable_text(html):
    if not isinstance(html, str):
        return None
    text = re.sub(r'(?is)<script[^>]*>.*?</script>', ' ', html)
    text = re.sub(r'(?is)<style[^>]*>.*?</style>', ' ', text)
    text = re.sub(r'(?is)<noscript[^>]*>.*?</noscript>', ' ', text)
    text = re.sub(r'(?s)<[^>]+>', ' ', text)
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 200:
        return None
    return text[:60000]

def _read_url_direct(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = _DIRECT_SESSION.get(url, headers=headers, timeout=_DIRECT_TIMEOUT)
        if response.status_code != 200:
            return None
        content_type = response.headers.get("Content-Type", "")
        body = response.text
        if "html" in content_type.lower():
            return _extract_readable_text(body)
        text = body.strip()
        return text[:60000] if len(text) >= 200 else None
    except Exception:
        return None

def read_url_with_jina(url):
    """
    Read webpage content using Jina Reader.
    """
    jina_url = f"https://r.jina.ai/{url}"
    try:
        logger.info(f"正在使用 Jina Reader 读取 URL: {url}")
        
        # Set headers to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = _JINA_SESSION.get(jina_url, headers=headers, timeout=_JINA_TIMEOUT)
        
        if response.status_code != 200:
            logger.warning(f"Jina Reader 返回非 200 状态码: {response.status_code}, URL: {url}")
            fallback_content = _read_url_direct(url)
            if fallback_content:
                logger.info(f"Jina 返回非200，使用直连读取成功: {url}")
            return fallback_content
            
        response.raise_for_status()
        content = response.text
        
        # Check for specific error messages in content
        if "451 Unavailable For Legal Reasons" in content:
            logger.warning(f"Jina Reader 无法读取 (法律原因): {url}")
            fallback_content = _read_url_direct(url)
            if fallback_content:
                logger.info(f"Jina 法律限制，使用直连读取成功: {url}")
            return fallback_content
            
        logger.debug(f"Jina Reader 读取成功，内容长度: {len(content)}")
        return content
        
    except requests.exceptions.Timeout:
        logger.warning(f"Jina Reader 请求超时 ({url})")
        fallback_content = _read_url_direct(url)
        if fallback_content:
            logger.info(f"Jina 超时，使用直连读取成功: {url}")
        return fallback_content
    except Exception as e:
        error_msg = f"Jina Reader 读取失败 ({url}): {e}"
        logger.warning(error_msg)
        fallback_content = _read_url_direct(url)
        if fallback_content:
            logger.info(f"Jina 失败，使用直连读取成功: {url}")
        return fallback_content
