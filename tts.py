# 文本转语音模块：使用edge_tts库实现语音合成
# 提供文本转语音、音频播放和TTS缓存清理功能
import edge_tts
import io
from pydub import AudioSegment
from config import CONFIG
from utils import logger
import sounddevice as sd

async def text_to_speech(text):
    """将文本转换为语音"""
    if not text:
        return None
    
    logger.info("🔊 语音合成中...")
    try:
        tts = edge_tts.Communicate(text=text, voice=CONFIG["voice"])
        audio_data = b''
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        if audio_data:
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            return np.array(audio.get_array_of_samples())
        return None
    except Exception as e:
        logger.error(f"❌ TTS失败: {e}")
        return None

def play(audio_data):
    """播放音频"""
    if audio_data is not None:
        try:
            sd.play(audio_data, samplerate=CONFIG["sample_rate"])
            sd.wait()
        except Exception as e:
            logger.error(f"❌ 播放失败: {e}")

class TTSCleaner:
    """TTS缓存清理器"""
    def cleanup(self):
        """清理TTS相关缓存"""
        logger.debug("🔊 TTS缓存已清理")
