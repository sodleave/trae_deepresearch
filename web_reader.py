import requests
from config import logger

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
        
        response = requests.get(jina_url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            logger.warning(f"Jina Reader 返回非 200 状态码: {response.status_code}, URL: {url}")
            return None
            
        response.raise_for_status()
        content = response.text
        
        # Check for specific error messages in content
        if "451 Unavailable For Legal Reasons" in content:
            logger.warning(f"Jina Reader 无法读取 (法律原因): {url}")
            return None
            
        logger.debug(f"Jina Reader 读取成功，内容长度: {len(content)}")
        return content
        
    except requests.exceptions.Timeout:
        logger.warning(f"Jina Reader 请求超时 ({url})")
        return None
    except Exception as e:
        error_msg = f"Jina Reader 读取失败 ({url}): {e}"
        logger.warning(error_msg)
        return None
