from concurrent.futures import ThreadPoolExecutor
from config import TAVILY_API_KEY, LLM_API_KEY, logger
from search_service import search_tavily
from llm_service import decompose_question, summarize_with_llm, extract_key_info, select_relevant_urls, analyze_search_results, plan_next_step, generate_final_answer
from web_reader import read_url_with_jina
from cache_manager import cache_manager

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
        
        original_question = user_input
        current_question = user_input
        confirmed_info = []
        iteration = 0
        max_iterations = 3
        final_answer = None

        while iteration < max_iterations:
            iteration += 1
            print(f"\n>>> 第 {iteration} 轮探索 <<<")
            
            # 1. Planning agent: Analyze original question and confirmed info
            print("正在规划下一步搜索方向...")
            plan = plan_next_step(original_question, confirmed_info)
            
            if not plan:
                print("规划出错，尝试直接搜索原始问题...")
                current_question = original_question
            elif plan.get("is_fully_answered"):
                final_answer = plan.get("final_answer")
                print(">>> 信息已收集完整，可以回答原始问题！ <<<")
                break
            else:
                missing_analysis = plan.get("missing_info_analysis", "")
                current_question = plan.get("next_question", original_question)
                print(f"分析: {missing_analysis}")
                print(f"当前聚焦问题: {current_question}")
            
            print(f"正在分析并拆解当前问题: {current_question} ...")
            sub_queries = decompose_question(current_question)
            
            # Ensure current question is in search queries
            search_queries = list(set([current_question] + sub_queries))
            print(f"生成的搜索查询: {search_queries}")
            logger.info(f"第 {iteration} 轮搜索查询: {search_queries}")
            
            print(f"正在并发搜索 {len(search_queries)} 个查询 ...")
            
            all_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Concurrent search
                futures = [executor.submit(search_tavily, q) for q in search_queries]
                for future in futures:
                    result = future.result()
                    if result:
                        all_results.append(result)
            
            extracted_contents = []
            
            if all_results:
                print("搜索完成，正在进行深度阅读和信息提取...")
                
                # Intelligent URL selection using LLM
                target_limit = 5
                urls_to_read = select_relevant_urls(current_question, all_results, limit=target_limit)
                
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
                    urls_to_read = urls[:10] # Get more for retry

                print(f"已筛选出 {len(urls_to_read)} 个候选网页，目标阅读 {target_limit} 个...")
                
                # Read and extract with retry mechanism and caching
                for url in urls_to_read:
                    # Check if we have enough content
                    if len(extracted_contents) >= target_limit:
                        break
                        
                    print(f"正在处理: {url}")
                    
                    # Check cache first
                    content = cache_manager.get(url)
                    
                    if content:
                        print(f"  -> 命中本地缓存")
                    else:
                        # Fetch from web
                        content = read_url_with_jina(url)
                        if content:
                            # Cache the successful read
                            cache_manager.set(url, content)
                    
                    if content:
                        print(f"  -> 正在提取关键信息 ({len(content)} 字符)...")
                        # Use current_question for extraction relevance
                        info = extract_key_info(current_question, content)
                        if info and "无相关信息" not in info:
                            extracted_contents.append({
                                "title": f"深度阅读: {url}",
                                "url": url,
                                "content": info
                            })
                            print(f"  -> 提取成功 (当前进度: {len(extracted_contents)}/{target_limit})")
                        else:
                            print("  -> 内容与问题相关度低，跳过")
                    else:
                        print("  -> 读取失败，尝试下一个候选网页...")
            else:
                print("本轮搜索未返回结果。")

            # Prepare data for analysis
            analysis_input = []
            if extracted_contents:
                print(f"成功从 {len(extracted_contents)} 个网页提取信息，正在分析...")
                analysis_input = [{"results": extracted_contents}]
            elif all_results:
                print("深度阅读未能提取有效信息，使用搜索摘要进行分析...")
                analysis_input = all_results
            else:
                print("未能获取任何信息，尝试下一轮...")
                if iteration == max_iterations:
                     break
                continue

            # Analyze results against the CURRENT simplified question
            analysis = analyze_search_results(current_question, analysis_input)
            
            if not analysis:
                print("分析过程出错，尝试继续...")
                continue
                
            if analysis.get("is_answered"):
                print(">>> 当前聚焦问题已得到解答 <<<")
                new_confirmed = analysis.get("new_confirmed_info", [])
                
                if new_confirmed:
                    confirmed_info.extend(new_confirmed)
                    # Simple deduplication
                    confirmed_info = list(set(confirmed_info))
                    print(f"新增确认信息 ({len(new_confirmed)}条):")
                    for info in new_confirmed:
                        print(f"  - {info}")
            else:
                print(">>> 当前搜索未能解答聚焦问题，将在下一轮重新规划 <<<")
        
        # After max iterations, check one last time if we can answer
        if not final_answer:
            print("\n达到最大轮次，正在生成最终简明回答...")
            final_answer = generate_final_answer(original_question, confirmed_info)
        
        if final_answer:
            print("\n=== 深度回答 ===")
            print(final_answer)
            print("================")
        else:
            print("\n=== 最终总结 (基于已收集信息) ===")
            if confirmed_info:
                print("已确认的信息:")
                for info in confirmed_info:
                    print(f"- {info}")
            else:
                print("未能收集到有效信息。")
            print("================")

if __name__ == "__main__":
    main()
