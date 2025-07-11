import requests
import threading
import queue
import sounddevice as sd
from openai import OpenAI
import re
import time

# ——— 設定 ———
BASE_URL = "http://127.0.0.1:50022"
SPEAKER_ID = 0
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 1024
API_KEY = ""


# グローバルキューとセッション
pcm_q = queue.Queue(maxsize=50)
text_q = queue.Queue()
sess = requests.Session()
client = OpenAI(api_key=API_KEY)

def get_chat_response(user_input):
    """ChatGPTからストリーミングレスポンスを取得し、文単位でキューに送信"""
    buffer = ""

    user_input += " 語尾を「ずんだもん」にして。また英語で端的に答えて"
    
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}],
        max_tokens=200,
        temperature=0.5,
        stream=True
    )
    
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        buffer += content
        
        # 文の区切りを検出（。！？で終わる）
        sentences = re.split(r'([。！？])', buffer)
        
        # 完全な文があればキューに送信
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                complete_sentence = sentences[i] + sentences[i+1]
                if complete_sentence.strip():
                    text_q.put(complete_sentence.strip())
        
        # 未完成の文は buffer に残す
        buffer = sentences[-1] if len(sentences) % 2 == 1 else ""
    
    # 残りのテキストを処理
    if buffer.strip():
        text_q.put(buffer.strip())
    
    text_q.put(None)  # 終了マーカー

def synthesize_audio():
    """テキストキューから音声合成してPCMキューに送信"""
    while True:
        text = text_q.get()
        if text is None:
            break
            
        try:
            # Audio Query
            r1 = sess.post(f"{BASE_URL}/audio_query",
                          params={"text": text, "speaker": SPEAKER_ID})
            r1.raise_for_status()
            
            # Synthesis
            r2 = sess.post(f"{BASE_URL}/synthesis",
                          params={"speaker": SPEAKER_ID},
                          json=r1.json(),
                          stream=True)
            r2.raise_for_status()
            
            # WAVヘッダーをスキップしてPCMデータを送信
            header_skipped = False
            for chunk in r2.iter_content(chunk_size=CHUNK_SIZE):
                if not header_skipped:
                    if len(chunk) > 44:
                        pcm_q.put(chunk[44:])
                    header_skipped = True
                else:
                    pcm_q.put(chunk)
                    
        except Exception as e:
            print(f"音声合成エラー: {e}")
    
    pcm_q.put(None)  # 終了マーカー

def play_audio():
    """PCMキューから音声再生"""
    with sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    ) as stream:
        while True:
            pcm = pcm_q.get()
            if pcm is None:
                break
            stream.write(pcm)

def voice_chat(user_input):
    """メイン処理：テキスト→音声のパイプライン"""
    print(f"ユーザー: {user_input}")
    print("AI: ", end="", flush=True)
    
    # 音声再生スレッド開始
    play_thread = threading.Thread(target=play_audio, daemon=True)
    play_thread.start()
    
    # 音声合成スレッド開始
    synth_thread = threading.Thread(target=synthesize_audio, daemon=True)
    synth_thread.start()
    
    # ChatGPT応答取得（メインスレッド）
    get_chat_response(user_input)
    
    # 全スレッド終了を待機
    synth_thread.join()
    play_thread.join()
    
    print("\n音声再生完了")

if __name__ == "__main__":
    while True:
        user_input = input("\n質問を入力してください (終了: quit): ")
        if user_input.lower() == 'quit':
            break
        
        # キューをクリア
        while not text_q.empty():
            text_q.get()
        while not pcm_q.empty():
            pcm_q.get()
            
        voice_chat(user_input)