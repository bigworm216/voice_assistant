# 大语言模型交互模块：封装百度API调用逻辑
# 提供API请求、缓存管理和本地备用响应功能
import aiohttp
import json
from config import CONFIG
from utils import logger
async def ask(query: str) -> str:
    """调用百度API"""
    if not query:
        return ""

    logger.info("📡 请求API中...")
    try:
        payload = {
            "model": CONFIG["baidu_model"],
            "messages": [{"role": "user", "content": query}],
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "penalty_score": 1.0
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            async with session.post(
                CONFIG["baidu_api_url"],
                headers={
                    "Authorization": f"Bearer {CONFIG['baidu_api_key']}",
                    "Content-Type": "application/json"
                },
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ API错误 (状态码: {response.status})")
                    return f"API错误: {response.status}"

                data = await response.json()
                return data.get("result", "无法解析API响应")

    except Exception as e:
        logger.error(f"❌ 请求失败: {type(e).__name__}")
        return "服务异常"
