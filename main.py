#!/usr/bin/env python3
# 主程序模块：协调各组件工作流程
# 实现语音助手的主循环，包括唤醒词监听、对话处理和资源清理
import asyncio
from pathlib import Path
from config import CONFIG
from utils import logger
import sounddevice as sd
import numpy as np
from stt import load_model, transcribe  # ← 新增
from llm import ask
from tts import speak
from utils import MemoryManager, AudioBufferCleaner, WhisperModelCleaner
from wakeword import WakeWordDetector
import os
import resource
import sounddevice as sd
from faster_whisper import WhisperModel
from .audio import AudioManager
from .wakeword import WakeWordDetector
from .stt import WhisperOptimizer
from .utils import (
    MemoryManager, AudioBufferCleaner, WhisperModelCleaner, TTSCleaner,
    structured_logger, cache_manager, state
)

class VoiceAssistant:
    _http_session = None

    def __init__(self):
        """初始化语音助手"""
        load_model()  # ← 改为直接调用 stt.py 函数

    async def init_async(self):
        """异步初始化HTTP会话"""
        if VoiceAssistant._http_session is None:
            import aiohttp
            VoiceAssistant._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=aiohttp.TCPConnector(limit_per_host=3)
            )
            logger.info("🌐 HTTP会话已创建")

    def record_audio(self) -> np.ndarray:
        """录制音频"""
        logger.info(f"🎤 录音中（{CONFIG['duration']}秒）...")
        audio = sd.rec(
            int(CONFIG["sample_rate"] * CONFIG["duration"]),
            samplerate=CONFIG["sample_rate"],
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        return audio

    async def run(self):
        """主运行循环"""
        try:
            audio = self.record_audio()
            if audio is None:
                return
            query = transcribe(audio)  # ← 直接调用 stt.py 函数
            if not query:
                logger.warning("⚠️ 未识别到有效输入")
                return
            logger.info(f"👂 识别结果: {query}")
            reply = await ask(query)
            logger.info(f"🤖 百度回复: {reply}")
            raw = await speak(reply)
            from audio import play
            play(raw)
        except Exception as e:
            logger.error(f"💥 系统错误: {e}")
        finally:
            if VoiceAssistant._http_session:
                import aiohttp
                await VoiceAssistant._http_session.close()
                logger.info("🌐 HTTP会话已关闭")


# 修复导入错误和重复导入
import asyncio
import os
from pathlib import Path
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from config import CONFIG
from audio import AudioManager
from wakeword import WakeWordDetector
from stt import WhisperOptimizer
from tts import text_to_speech
from llm import call_baidu_api_optimized, get_local_fallback_response
from utils import (
    logger, structured_logger, state, cache_manager,
    MemoryManager, AudioBufferCleaner, WhisperModelCleaner, TTSCleaner
)

# 移除未使用的VoiceAssistant类定义

async def main():
    ""主循环""
    logger.info("🚀 语音助手启动")
    logger.info(f"📋 唤醒词: {', '.join(CONFIG['wake_words'])}")
    
    # 初始化组件
    model = initialize_models()
    device_id = initialize_audio()
    audio_manager = AudioManager(device_id, CONFIG["sample_rate"], CONFIG["audio_chunk_size"])
    wake_detector = WakeWordDetector()
    whisper_optimizer = WhisperOptimizer(model)
    memory_manager = MemoryManager()
    
    # 初始化清理器
    audio_cleaner = AudioBufferCleaner(audio_manager)
    whisper_cleaner = WhisperModelCleaner(whisper_optimizer)
    tts_cleaner = TTSCleaner()
    
    # 注册清理回调
    memory_manager.add_cleanup_callback(audio_cleaner.cleanup)
    memory_manager.add_cleanup_callback(whisper_cleaner.cleanup)
    memory_manager.add_cleanup_callback(tts_cleaner.cleanup)
    memory_manager.add_cleanup_callback(cache_manager.clear_expired)
    
    try:
        # 添加缺失的HTTP会话初始化
        await initialize_http_session()
        
        with audio_session(audio_manager):
            while True:
                # 监听唤醒词
                logger.info("👂 监听唤醒词...")
                audio = audio_manager.record_for_duration(CONFIG["listen_duration"])
                
                if audio is None:
                    continue
                
                # 检测唤醒词
                is_wake_word, score = wake_detector.detect(audio)
                if not is_wake_word:
                    logger.debug(f"未检测到唤醒词 (置信度: {score:.2f})")
                    continue
                
                logger.info(f"🎯 检测到唤醒词 (置信度: {score:.2f})")
                
                # 对话阶段
                logger.info("💬 进入对话模式...")
                audio = audio_manager.record_until_silence(
                    CONFIG["silence_duration"],
                    CONFIG["max_conversation_time"]
                )
                
                if audio is None:
                    logger.warning("⚠️ 对话录音失败")
                    continue
                
                query = whisper_optimizer.transcribe_conversation_optimized(audio)
                if not query:
                    logger.warning("⚠️ 未识别到语音")
                    continue
                
                logger.info(f"🗣️ 识别结果: {query}")
                
                # 本地备用响应检查
                response = get_local_fallback_response(query)
                if not response:
                    # 调用LLM
                    response = await call_baidu_api_optimized(query)
                
                if response:
                    logger.info(f"🤖 回复: {response[:50]}...")
                    await text_to_speech(response)
                
                # 性能统计
                whisper_optimizer.print_performance_stats()
                
                # 定期清理
                if state.conversation_count % CONFIG["cleanup_interval"] == 0:
                    memory_manager.force_cleanup("定期清理")
                
                state.conversation_count += 1
                await asyncio.sleep(0.5)
                
    except KeyboardInterrupt:
        logger.info("👋 用户中断，退出程序")
    except Exception as e:
        logger.error(f"主循环错误: {e}", exc_info=True)
    finally:
        memory_manager.force_cleanup("程序退出清理")
        await close_http_session()
        logger.info("👋 语音助手已退出")

    logger.info(f"👂 识别结果: {query}")
    reply = await ask(query)
    logger.info(f"🤖 百度回复: {reply}")
    raw = await speak(reply)
    from audio import play
    play(raw)

    await VoiceAssistant._http_session.close()

    logger.info("🌐 HTTP会话已关闭")

# 添加模型和音频初始化函数
def initialize_models():
    ""针对硬件优化的模型初始化""
    structured_logger.log_operation("Whisper模型初始化", "开始")
    
    try:
        # CPU优化设置
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        
        model = WhisperModel(
            model_size_or_path=CONFIG["model_path"],
            device="cpu",
            compute_type=CONFIG["compute_type"],
            local_files_only=True,
            cpu_threads=2,
            num_workers=1
        )
        structured_logger.log_success("Whisper模型初始化完成")
        return model
    except Exception as e:
        structured_logger.log_error("Whisper模型初始化失败", str(e))
        raise


def initialize_audio():
    ""音频设备初始化""
    structured_logger.log_operation("音频设备初始化", "开始")
    
    try:
        # 重置音频子系统
        sd._terminate()
        sd._initialize()
        
        devices = sd.query_devices()
        usb_devices = [i for i, device in enumerate(devices) if device['max_input_channels'] > 0 and 'USB' in str(device).upper()]
        
        if usb_devices:
            device_id = usb_devices[0]
            audio_logger.info(f"使用USB音频设备: {device_id}")
        else:
            device_id = sd.default.device[0]
            audio_logger.info(f"使用默认音频设备: {device_id}")
        
        # 测试音频设备
        with sd.InputStream(samplerate=CONFIG["sample_rate"], channels=1, device=device_id):
            pass
        
        structured_logger.log_success("音频设备初始化完成")
        return device_id
    except Exception as e:
        structured_logger.log_error("音频设备初始化失败", str(e))
        raise

if __name__ == "__main__":
    import os, sys
    if os.geteuid() == 0:
        logger.warning("⚠️ 不建议以root用户运行")
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
