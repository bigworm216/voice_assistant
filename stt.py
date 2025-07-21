# 语音转文本模块：使用Faster Whisper模型实现语音识别
# 集成自适应噪声检测和动态阈值调整功能
from faster_whisper import WhisperModel
from pathlib import Path
# 添加缺失的导入
import time
import numpy as np
from config import CONFIG
from utils import whisper_logger, system_logger

_model = None

# 保留原有的load_model和transcribe函数

def load_model():
    global _model
    if _model is None:
        whisper_logger.info("🔄 正在加载本地模型...")
        _model = WhisperModel(
            CONFIG["local_model_path"],
            device="cpu",
            compute_type=CONFIG["compute_type"],
            cpu_threads=2,
            local_files_only=True
        )
    return _model


def transcribe(audio) -> str:
    model = load_model()
    segments, _ = model.transcribe(audio, beam_size=5, language="zh")
    return "".join(s.text for s in segments).strip()

class WhisperOptimizer:
    ""Whisper优化器 - 支持openwakeword和whisper双模式""
    
    def __init__(self, whisper_model):
        self.whisper_model = whisper_model
        self.wake_word_detector = WakeWordDetector() if CONFIG["wake_word_mode"] == "openwakeword" else None
        
        # 性能统计
        self.timing_stats = {
            "wake_word_calls": 0,\
            "conversation_calls": 0,
            "total_wake_word_time": 0.0,
            "total_conversation_time": 0.0
        }
        
        # 背景噪声检测
        self.background_noise_samples = []
        self.background_noise_db = -70
        self.noise_sample_count = 0
        self.noise_update_interval = CONFIG["adaptive_noise"]["update_interval"]
        self.current_threshold = CONFIG["volume_threshold"]
        
        system_logger.info(f"🎯 唤醒模式: {CONFIG['wake_word_mode']}")
    
    def detect_wake_word_optimized(self, audio):
        start_time = time.time()
        
        # 测量背景噪声（如果启用）
        if CONFIG["adaptive_noise"]["enabled"]:
            self.measure_background_noise(audio)
        
        try:
            # 使用openwakeword检测
            detected, score = self.wake_word_detector.detect(audio)
            self._update_timing_stats("wake_word", time.time() - start_time)
            return detected, "wake_word_detected" if detected else ""
            
        except Exception as e:
            whisper_logger.error(f"唤醒词检测失败: {e}")
            return False, ""
    
    def transcribe_conversation_optimized(self, audio):
        start_time = time.time()
        
        # 测量背景噪声（如果启用）
        if CONFIG["adaptive_noise"]["enabled"]:
            self.measure_background_noise(audio)
        
        try:
            # 使用自适应参数或固定参数
            if CONFIG["adaptive_noise"]["enabled"]:
                params = self.get_adaptive_params("conversation")
            else:
                params = CONFIG["whisper_params"]["conversation"]
            
            segments, _ = self.whisper_model.transcribe(
                audio,
                **params
            )
            
            text = "".join(seg.text for seg in segments).strip()
            
            self._update_timing_stats("conversation", time.time() - start_time)
            return text
            
        except Exception as e:
            whisper_logger.error(f"对话转录失败: {e}")
            return ""
    
    # 添加缺失的方法
    def _update_timing_stats(self, call_type, duration):
        if call_type == "wake_word":
            self.timing_stats["wake_word_calls"] += 1
            self.timing_stats["total_wake_word_time"] += duration
        else:
            self.timing_stats["conversation_calls"] += 1
            self.timing_stats["total_conversation_time"] += duration
        
        whisper_logger.debug(f"⏱️ {call_type} 转录耗时: {duration:.3f}s")
    
    def print_performance_stats(self):
        wake_calls = self.timing_stats["wake_word_calls"]
        conv_calls = self.timing_stats["conversation_calls"]
        
        if wake_calls > 0:
            avg_wake_time = self.timing_stats["total_wake_word_time"] / wake_calls
            whisper_logger.info(f"🤖 唤醒词检测: {wake_calls} 次，平均耗时: {avg_wake_time:.3f}s")
        
        if conv_calls > 0:
            avg_conv_time = self.timing_stats["total_conversation_time"] / conv_calls
            whisper_logger.info(f"🤖 对话转录: {conv_calls} 次，平均耗时: {avg_conv_time:.3f}s")
        
        # 打印背景噪声信息
        if CONFIG["adaptive_noise"]["enabled"]:
            status = self.get_adaptive_status()
            if isinstance(status, dict):
                whisper_logger.info(f"🔊 自适应噪声检测: {status['noise_level']}环境, {status['background_noise_db']:.1f} dB, 阈值: {status['current_threshold']:.2f}")
            else:
                whisper_logger.info(status)
        else:
            whisper_logger.info(f"🔊 当前背景噪声: {self.background_noise_db:.1f} dB")
    
    def get_adaptive_params(self, params_type: str = "wake_word"):
        ""获取自适应参数""
        base_params = CONFIG["whisper_params"][params_type].copy()
        
        # 动态调整no_speech_threshold
        adaptive_threshold = self.adaptive_no_speech_threshold()
        base_params["no_speech_threshold"] = adaptive_threshold
        
        return base_params
    
    def get_adaptive_status(self):
        ""获取自适应状态信息""
        if not CONFIG["adaptive_noise"]["enabled"]:
            return "自适应噪声检测已禁用"
        
        noise_config = CONFIG["adaptive_noise"]
        current_threshold = self.adaptive_no_speech_threshold()
        
        status = {
            "enabled": True,
            "background_noise_db": self.background_noise_db,
            "current_threshold": current_threshold,
            "sample_count": len(self.background_noise_samples),
            "noise_level": "未知"
        }
        
        # 确定噪声水平
        thresholds = noise_config["noise_thresholds"]
        if self.background_noise_db > thresholds["medium"]:
            status["noise_level"] = "嘈杂"
        elif self.background_noise_db > thresholds["quiet"]:
            status["noise_level"] = "中等"
        else:
            status["noise_level"] = "安静"
        
        return status
    
    def measure_background_noise(self, audio):
        
        try:
            # 计算音频的RMS值
            rms = np.sqrt(np.mean(audio**2))
            
            # 转换为分贝
            if rms > 0:
                db = 20 * np.log10(rms)
            else:
                db = -100  # 静音
            
            # 添加到样本列表
            self.background_noise_samples.append(db)
            self.noise_sample_count += 1
            
            # 保持样本数量在合理范围内
            if len(self.background_noise_samples) > 50:
                self.background_noise_samples = self.background_noise_samples[-30:]
            
            # 定期更新背景噪声水平
            if self.noise_sample_count % self.noise_update_interval == 0:
                # 使用中位数作为背景噪声水平，避免异常值影响
                self.background_noise_db = np.median(self.background_noise_samples)
                whisper_logger.debug(f"🔊 背景噪声更新: {self.background_noise_db:.1f} dB (样本数: {len(self.background_noise_samples)})")
            
            return db
            
        except Exception as e:
            whisper_logger.error(f"背景噪声测量失败: {e}")
            return -70  # 默认值
    
    def adaptive_no_speech_threshold(self):
        ""根据背景噪声动态调整阈值"""
        noise_config = CONFIG["adaptive_noise"]
        thresholds = noise_config["noise_thresholds"]
        values = noise_config["threshold_values"]
        
        if self.background_noise_db > thresholds["medium"]:
            threshold = values["noisy"]  # 嘈杂环境提高阈值
            whisper_logger.debug(f"🔊 嘈杂环境，使用高阈值: {threshold}")
        elif self.background_noise_db > thresholds["quiet"]:
            threshold = values["medium"]  # 中等噪声环境
            whisper_logger.debug(f"🔊 中等噪声环境，使用中等阈值: {threshold}")
        else:
            threshold = values["quiet"]  # 安静环境降低阈值
            whisper_logger.debug(f"🔊 安静环境，使用低阈值: {threshold}")
        
        return threshold
}
