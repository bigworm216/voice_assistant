# 唤醒词检测模块：基于openwakeword实现低功耗唤醒词识别
# 提供唤醒词检测、模型管理和置信度评分功能
from openwakeword import Model
import numpy as np
from config import CONFIG
import logging

error_logger = logging.getLogger('error')
system_logger = logging.getLogger('system')
class WakeWordDetector:
    def __init__(self):
        self.config = CONFIG["openwakeword"]
        try:
            self.model = Model(
                wakeword_models=[self.config["model_path"]],
                inference_framework=self.config["inference_framework"],
                enable_speex_noise_suppression=self.config["enable_speex_noise_suppression"]
            )
            system_logger.info("✅ openwakeword模型加载成功")
        except Exception as e:
            error_logger.error(f"❌ openwakeword加载失败: {e}")
            self.model = None
    
    def detect(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        """
        检测唤醒词
        
        Args:
            audio_chunk: 音频数据
            
        Returns:
            (是否唤醒, 置信度分数)
        """
        if self.model is None:
            return False, 0.0
            
        try:
            # 确保音频数据格式正确
            if len(audio_chunk) != self.config["buffer_size"]:
                return False, 0.0
                
            prediction = self.model.predict(audio_chunk)
            
            for model_name, score in prediction.items():
                if score > self.config["threshold"]:
                    return True, score
                    
            return False, max(prediction.values()) if prediction else 0.0
            
        except Exception as e:
            error_logger.error(f"openwakeword检测错误: {e}")
            return False, 0.0
    
    def get_model_info(self) -> dict:
        ""获取模型信息"""
        return {
            "loaded": self.model is not None,
            "models": list(self.model.models.keys()) if self.model else [],
            "framework": self.config["inference_framework"]
        }