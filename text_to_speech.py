import requests
import threading
import queue
import sounddevice as sd

# ——— 設定 ———
BASE_URL    = "http://127.0.0.1:50022"
SPEAKER_ID  = 0
SAMPLE_RATE = 24000     # Engine に合わせる
CHANNELS    = 1
CHUNK_SIZE  = 1024      # 小さめに

# 分割済みの文リスト
TEXT = "ずんだもんなのだ!!献上品を送るのだ"
sentences = [s.strip() + "。" for s in TEXT.split("。") if s.strip()]

# PCM チャンクを流すキュー
pcm_q = queue.Queue(maxsize=50)

# セッション再利用
sess = requests.Session()

# ——— Producer: 合成 → PCMチャンクをキューに投入 ———
def producer():
    for sent in sentences:
        # 1) Audio Query
        r1 = sess.post(f"{BASE_URL}/audio_query",
                       params={"text": sent, "speaker": SPEAKER_ID})
        r1.raise_for_status()
        aq = r1.json()

        # 2) Synthesis (WAV bytes ストリーミング)
        r2 = sess.post(f"{BASE_URL}/synthesis",
                       params={"speaker": SPEAKER_ID},
                       json=aq,
                       stream=True)
        r2.raise_for_status()

        # ヘッダ部を自前でスキップ
        it = r2.iter_content(chunk_size=CHUNK_SIZE)
        header_skipped = False
        for chunk in it:
            if not header_skipped:
                # ヘッダがchunkより短ければ、差分を再利用
                if len(chunk) > 44:
                    pcm_q.put(chunk[44:])
                # chunk == 44 なら無視、<44 はVoiceVoxでは起こらない前提
                header_skipped = True
            else:
                pcm_q.put(chunk)

    # 全文合成完了を示すマーカー
    pcm_q.put(None)

# ——— Consumer: キューから取り出して即再生 ———
def consumer():
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

if __name__ == "__main__":
    # 1) consumer スレッド起動
    t_cons = threading.Thread(target=consumer, daemon=True)
    t_cons.start()

    # 2) producer 実行（メインスレッドでもOK）
    producer()

    # 3) consumer の終了を待つ
    t_cons.join()
