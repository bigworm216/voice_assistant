# 工具函数模块：提供通用辅助功能
# 包含日志配置、内存管理、缓存清理和性能统计工具类
import logging          # ← 补上这行
import os               # 原文件头部也在这里

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VoiceAssistant")

# 强制使用CPU模式
os.environ["CT2_FORCE_CPU_ONLY"] = "1"

import psutil
import gc
import time
from typing import Dict, List, Callable
import logging

memory_logger = logging.getLogger('memory')

class MemoryManager:
    ""内存管理器"""
    
    def __init__(self):
        self.memory_threshold = CONFIG["memory_threshold_mb"]
        self.cleanup_callbacks = []
        self.memory_history = []
    
    def get_memory_usage(self) -> Dict[str, float]:
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except Exception as e:
            memory_logger.error(f"获取内存信息失败: {e}")
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}
    
    def force_cleanup(self, reason: str = "定期清理"):
        memory_logger.info(f"🧹 开始内存清理: {reason}")
        
        before_memory = self.get_memory_usage()
        
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                memory_logger.error(f"清理回调失败: {e}")
        
        collected = gc.collect()
        after_memory = self.get_memory_usage()
        memory_freed = before_memory["rss_mb"] - after_memory["rss_mb"]
        
        memory_logger.info(f"✅ 内存清理完成: 释放 {memory_freed:.1f} MB, 回收 {collected} 个对象")
        
        self.memory_history.append({
            "timestamp": time.time(),
            "before_mb": before_memory["rss_mb"],
            "after_mb": after_memory["rss_mb"],
            "freed_mb": memory_freed,
            "collected_objects": collected,
            "reason": reason
        })
        
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-50:]
    
    def add_cleanup_callback(self, callback: Callable):
        self.cleanup_callbacks.append(callback)
    
    def print_stats(self):
        if not self.memory_history:
            return
        
        total_cleanups = len(self.memory_history)
        total_freed = sum(h["freed_mb"] for h in self.memory_history)
        avg_freed = total_freed / total_cleanups
        
        memory_logger.info(f"📊 内存清理统计: 总次数 {total_cleanups}, 总释放 {total_freed:.1f} MB, 平均 {avg_freed:.1f} MB")

class AudioBufferCleaner:
    def __init__(self, audio_manager):
        self.audio_manager = audio_manager
    
    def cleanup(self):
        try:
            with self.audio_manager.lock:
                self.audio_manager.audio_buffer.clear()
                if hasattr(self.audio_manager, '_temp_audio'):
                    del self.audio_manager._temp_audio
                    self.audio_manager._temp_audio = None
            audio_logger.debug("🎤 音频缓冲区已清理")
        except Exception as e:
            audio_logger.error(f"音频缓冲区清理失败: {e}")

class WhisperModelCleaner:
    def __init__(self, whisper_optimizer):
        self.whisper_optimizer = whisper_optimizer
        self.segment_cache = []
    
    def cleanup(self):
        try:
            self.segment_cache.clear()
            if hasattr(self.whisper_optimizer, 'performance_history'):
                self.whisper_optimizer.performance_history.clear()
            if hasattr(self.whisper_optimizer, 'timing_stats'):
                for key in list(self.whisper_optimizer.timing_stats.keys()):
                    if isinstance(self.whisper_optimizer.timing_stats[key], (list, dict)):
                        self.whisper_optimizer.timing_stats[key].clear()
            whisper_logger.debug("🤖 Whisper模型缓存已清理")
        except Exception as e:
            whisper_logger.error(f"Whisper模型清理失败: {e}")

# 添加缺失的日志器定义
class SimpleLogger:
    "简化日志器"
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter())
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str): self.logger.debug(message)
    def info(self, message: str): self.logger.info(message)
    def warning(self, message: str): self.logger.warning(message)
    def error(self, message: str): self.logger.error(message)
    def critical(self, message: str): self.logger.critical(message)

# 创建专用日志器
audio_logger = SimpleLogger("audio")
whisper_logger = SimpleLogger("whisper")
network_logger = SimpleLogger("network")
memory_logger = SimpleLogger("memory")
cache_logger = SimpleLogger("cache")
tts_logger = SimpleLogger("tts")
api_logger = SimpleLogger("api")
error_logger = SimpleLogger("error")
system_logger = SimpleLogger("system")
user_logger = SimpleLogger("user")
logger = SimpleLogger("main")

class StructuredLogger:
    ""结构化日志器""
    
    def __init__(self, base_logger: SimpleLogger):
        self.base_logger = base_logger
    
    def log_operation(self, operation: str, status: str = "开始", details: str = ""):
        status_icons = {"开始": "▶️", "完成": "✅", "失败": "❌", "跳过": "⏭️", "重试": "🔄"}
        icon = status_icons.get(status, "📝")
        message = f"{icon} {operation} {status}"
        if details: message += f" - {details}"
        self.base_logger.info(message)
    
    def log_performance(self, operation: str, duration: float):
        if duration < 1:
            duration_str = f"{duration*1000:.1f}毫秒"
        else:
            duration_str = f"{duration:.3f}秒"
        self.base_logger.info(f"⏱️ {operation} 耗时: {duration_str}")
