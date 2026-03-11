# 简易版 DeepResearch

这是一个最简版的 DeepResearch 实现，用于通过体感学习 DeepResearch 的演进过程。

## 功能
- 输入问题
- 调用 Tavily Search API 获取搜索结果
- 使用 LLM (OpenAI 兼容接口) 总结搜索结果并回答问题

## 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境变量**
   复制 `.env.example` 为 `.env` (如果尚未存在)，并填入你的 API Key：
   - `TAVILY_API_KEY`: Tavily 搜索 API Key (可在 https://tavily.com/ 获取)
   - `LLM_BASE_URL`: LLM API 基础地址 (例如 OpenAI 官方或兼容服务商)
   - `LLM_API_KEY`: LLM API Key
   - `LLM_MODEL`: 使用的模型名称 (默认 gpt-3.5-turbo)

3. **运行脚本**
   ```bash
   python deepresearch.py
   ```

## 文件结构
- `deepresearch.py`: 主程序脚本
- `.env`: 配置文件 (需手动填写 Key)
- `requirements.txt`: 依赖列表
