import sounddevice as sd
import numpy as np
import whisper
import scipy.signal
from scipy.signal import butter, filtfilt

# 設定
MIC_DEVICE = 5
SAMPLE_RATE = 44100
CHANNELS = 1
TARGET_RATE = 16000

# Whisperモデル読み込み（精度重視でbaseモデルを使用）
print("🔄 Whisperモデル読み込み中...")
model = whisper.load_model("base")  # 精度向上のためbaseモデル使用
print("✅ Whisperモデル読み込み完了")

vad = webrtcvad.Vad(2)

def preprocess_audio_fast(audio, sr=16000):
    # 1) DC+正規化+クリップ を一括
    mean = audio.mean()
    centered = ne.evaluate("audio - mean")
    rms = np.sqrt(np.mean(centered**2))
    if rms > 0:
        normed = ne.evaluate("centered * (0.1 / rms)")
    else:
        normed = centered
    clipped = np.clip(normed, -0.95, 0.95)
    
    # 2) SOSバンドパスフィルタ
    sos = butter(4, [80/sr*2, 7500/sr*2], btype='band', output='sos')
    filtered = sosfiltfilt(sos, clipped)
    
    # 3) VAD トリミング
    # 16bit PCM に変換
    pcm16 = (filtered * 32767).astype('int16').tobytes()
    frame_bytes = int(sr * 0.03) * 2
    voiced = []
    for off in range(0, len(pcm16) - frame_bytes + 1, frame_bytes):
        chunk = pcm16[off:off+frame_bytes]
        if vad.is_speech(chunk, sr):
            voiced.append(chunk)
    if voiced:
        pcm_out = b''.join(voiced)
        return np.frombuffer(pcm_out, dtype='int16').astype('float32') / 32767
    else:
        return filtered

def record_audio(duration=5.0):
    """音声録音（デフォルト5秒）"""
    duration = max(1.0, min(duration, 30.0))  # 1-30秒に制限
    
    print(f"🔴 {duration}秒間録音開始...")
    print("   ※ マイクに近づいて、はっきりと話してください")
    
    # 録音実行
    audio = sd.rec(int(duration * SAMPLE_RATE), 
                   samplerate=SAMPLE_RATE, 
                   channels=CHANNELS, 
                   dtype='float32', 
                   device=MIC_DEVICE)
    sd.wait()
    
    # モノラル化
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # 16kHzにリサンプリング
    if SAMPLE_RATE != TARGET_RATE:
        audio = scipy.signal.resample(audio, int(len(audio) * TARGET_RATE / SAMPLE_RATE))
    
    return audio.astype(np.float32)

def transcribe_audio(audio_data):
    """音声データを文字起こし"""
    if len(audio_data) == 0:
        return "音声データが空です"
    
    print("📝 文字起こし中...")
    
    # 音声前処理
    processed_audio = preprocess_audio(audio_data)
    
    # 音声レベルチェック
    rms = np.sqrt(np.mean(processed_audio**2))
    duration = len(processed_audio) / TARGET_RATE
    
    print(f"🔍 処理後音声: {duration:.1f}秒, RMS: {rms:.4f}")
    
    if rms < 0.001 or duration < 0.3:
        return "音声が検出されませんでした（音量を上げるか、より長く話してください）"
    
    # Whisperで文字起こし（精度向上のためのオプション）
    result = model.transcribe(
        processed_audio, 
        language="ja", 
        fp16=False,
        temperature=0.0,        # 決定的な出力（ランダム性を排除）
        beam_size=5,           # ビーム探索で精度向上
        best_of=5,             # 複数候補から最良を選択
        patience=1.0,          # より長い探索
        condition_on_previous_text=False,  # 前のテキストに依存しない
        initial_prompt="以下は日本語の音声です。",  # 日本語認識のヒント
        suppress_tokens=[-1]   # 不要なトークンを抑制
    )
    
    text = result["text"].strip()
    
    # 結果の品質チェック
    if not text:
        return "音声を認識できませんでした"
    
    return text

def main():
    try:
        print("🎙️ 高精度音声認識プログラム")
        print("=" * 40)
        
        # 録音時間の入力
        try:
            duration = float(input("録音時間（秒）[デフォルト5]: ") or "5")
        except ValueError:
            duration = 5.0
        
        # 録音と文字起こし
        audio = record_audio(duration)
        text = transcribe_audio(audio)
        
        print(f"\n✅ 認識結果: 「{text}」")
        
    except KeyboardInterrupt:
        print("\n中断されました")
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()