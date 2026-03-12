from concurrent.futures import ThreadPoolExecutor
from config import TAVILY_API_KEY, LLM_API_KEY, logger
from search_service import search_tavily
from llm_service import decompose_question, summarize_with_llm, extract_key_info, select_relevant_urls
from web_reader import read_url_with_jina

def main():
    print("=== 深度版 DeepResearch (输入 'quit' 或 'exit' 退出) ===")
    logger.info("程序启动")
    
    # Check configuration
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
        
        # Ensure original question is in search queries
        search_queries = list(set([user_input] + sub_queries))
        print(f"生成的搜索查询: {search_queries}")
        logger.info(f"最终去重后的搜索查询列表: {search_queries}")
        
        print(f"正在并发搜索 {len(search_queries)} 个查询 ...")
        
        all_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Concurrent search
            futures = [executor.submit(search_tavily, q) for q in search_queries]
            for future in futures:
                result = future.result()
                if result:
                    all_results.append(result)
        
        if all_results:
            print("搜索完成，正在进行深度阅读和信息提取...")
            
            # Intelligent URL selection using LLM
            urls_to_read = select_relevant_urls(user_input, all_results, limit=5)
            
            if not urls_to_read:
                print("未能筛选出有效的 URL，尝试使用默认前 5 个...")
                # Fallback to collecting all URLs
                urls = []
                for search_results in all_results:
                    if search_results and "results" in search_results:
                        for result in search_results["results"]:
                            url = result.get("url")
                            if url and url not in urls:
                                urls.append(url)
                urls_to_read = urls[:5]

            print(f"计划深入阅读 {len(urls_to_read)} 个精选网页...")
            
            extracted_contents = []
            
            # Read and extract
            for url in urls_to_read:
                print(f"正在阅读: {url}")
                content = read_url_with_jina(url)
                if content:
                    print(f"正在提取关键信息 ({len(content)} 字符)...")
                    info = extract_key_info(user_input, content)
                    if info and "无相关信息" not in info:
                        extracted_contents.append({
                            "title": f"深度阅读: {url}",
                            "url": url,
                            "content": info
                        })
            
            if extracted_contents:
                print(f"成功从 {len(extracted_contents)} 个网页提取信息，正在生成最终回答...")
                # Construct data for summarize_with_llm
                enhanced_results = [{"results": extracted_contents}]
                summary = summarize_with_llm(user_input, enhanced_results)
            else:
                print("深度阅读未能提取有效信息，回退到使用搜索摘要...")
                summary = summarize_with_llm(user_input, all_results)
            
            print("\n=== 深度回答 ===")
            print(summary)
            print("================")
        else:
            print("搜索未能获取结果，请重试。")
            logger.warning("所有搜索均未返回结果")

if __name__ == "__main__":
    main()
