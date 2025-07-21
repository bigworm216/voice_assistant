# 配置模块：存储项目所有配置参数
# 包含模型路径、API密钥、音频参数、唤醒词设置等全局配置
CONFIG = {
    "local_model_path": "/home/bigworm/.cache/faster-whisper/local-models/faster-whisper-small",
    "sample_rate": 16000,
    "duration": 5,
    "voice": "zh-CN-XiaoxiaoNeural",
    "compute_type": "int8",
    "baidu_api_key": "bce-v3/ALTAK-XPoPXbXixX22kiQRuPibn/66b2dfc146f90970ad7b439a92b59cd86c88926f",
    "baidu_api_url": "https:#qianfan.baidubce.com/v2/chat/completions",
    "baidu_model": "amv-m3s7wybzu6as",
    "memory_threshold": 75,
    "idle_timeout": 300,
    "max_retention": 600,
    
    # 添加缺失的配置项
    "listen_duration": 1.5,
    "silence_duration": 1.0,
    "max_conversation_time": 30,
    "audio_chunk_size": 0.05,
    "volume_threshold": 0.008,
    "cleanup_interval": 5,
    
    # 唤醒词配置
    "wake_words": ["你好小智"],  # 单一唤醒词
    "wake_word_confidence": 0.7,  # 唤醒词置信度阈值
    
    # openwakeword优化配置
    "openwakeword": {
        "model_path": "hey_janet",
        "inference_framework": "onnx",
        "threshold": 0.7,  # 提高唤醒阈值减少误唤醒
        "enable_speex_noise_suppression": True,
        "buffer_size": 1280,
        "sample_rate": 16000,
        "vad_threshold": 0.6,
        "min_activation_count": 3
    },
    
    # 新增Whisper参数配置
    "whisper_params": {
        "wake_word": {
            "beam_size": 1,
            "language": "zh",
            "vad_filter": True,
            "temperature": 0.0,
            "no_speech_threshold": 0.7,
        },
        "conversation": {
            "beam_size": 2,
            "language": "zh",
            "vad_filter": True,
            "temperature": 0.0,
            "no_speech_threshold": 0.6,
        }
    },
    
    # 网络配置
    "cache_expire": 3600,
    "max_retries": 3,
    
    # 自适应噪声检测配置
    "adaptive_noise": {
        "enabled": True,
        "update_interval": 10,
        "noise_thresholds": {
            "quiet": -70,
            "medium": -60,
            "noisy": -50
        },
        "threshold_values": {
            "quiet": 0.4,
            "medium": 0.6,
            "noisy": 0.7
        }
    },
    
    # 唤醒词检测模式
    "wake_word_mode": "openwakeword"
}
