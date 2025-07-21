# 音频处理模块：负责音频录制、播放和音频流管理
# 提供音频录制、停止、缓冲区清理等功能
# 完整实现AudioManager类
class AudioManager:
    ""长期存活音频流管理器""
    
    def __init__(self, device_id, sample_rate, chunk_size):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        self.is_active = False
        self.audio_buffer = []
        self.lock = threading.Lock()
    
    def start_stream(self):
        if self.stream is not None:
            return
        
        try:
            logger.info("🔧 启动长期存活音频流...")
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                device=self.device_id,
                blocksize=int(self.sample_rate * self.chunk_size),
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_active = True
            logger.info("✅ 音频流启动成功")
        except Exception as e:
            logger.error(f"❌ 音频流启动失败: {e}")
            raise
    
    def stop_stream(self):
        if self.stream is None:
            return
        
        try:
            logger.info("🛑 停止音频流...")
            self.is_active = False
            self.stream.stop()
            self.stream.close()
            self.stream = None
            logger.info("✅ 音频流已停止")
        except Exception as e:
            logger.error(f"❌ 停止音频流失败: {e}")
    
    def _audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"音频回调状态: {status}")
        
        if self.is_active and not state.is_speaking:
            audio_chunk = indata.astype(np.float32) / 32768.0
            audio_chunk = np.squeeze(audio_chunk)
            
            with self.lock:
                if state.is_recording:
                    self.audio_buffer.extend(audio_chunk)
    
    def start_recording(self):
        with self.lock:
            self.audio_buffer.clear()
            state.is_recording = True
            logger.info("🎤 开始录音...")
    
    def stop_recording(self):
        with self.lock:
            state.is_recording = False
            audio_data = np.array(self.audio_buffer) if self.audio_buffer else None
            self.audio_buffer.clear()
            
            if audio_data is not None and len(audio_data) > 0:
                duration = len(audio_data) / self.sample_rate
                logger.info(f"✅ 录音完成，时长: {duration:.2f}秒")
                return audio_data
            else:
                logger.warning("⚠️ 未录制到音频")
                return None
    
    def record_for_duration(self, duration):
        self.start_recording()
        time.sleep(duration)
        return self.stop_recording()
    
    def record_until_silence(self, silence_duration, max_duration):
        self.start_recording()
        
        silence_start = None
        start_time = time.time()
        
        while True:
            time.sleep(0.1)
            
            if time.time() - start_time > max_duration:
                logger.info("⏰ 达到最大录音时长")
                break
            
            with self.lock:
                if len(self.audio_buffer) > 0:
                    recent_audio = self.audio_buffer[-int(self.sample_rate/10):]
                    volume = np.sqrt(np.mean(np.array(recent_audio)**2))
                    
                    if volume > CONFIG["volume_threshold"]:
                        silence_start = None
                    else:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > silence_duration:
                            logger.info("🔇 检测到静音，停止录音")
                            break
        
        return self.stop_recording()

@contextlib.contextmanager
    def audio_session(audio_manager):
    ""音频会话管理器""
    try:
        audio_manager.start_stream()
        yield audio_manager
    finally:
        audio_manager.stop_stream()