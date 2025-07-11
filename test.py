txt_path = r"C:\Users\moriya_nishizawa\Desktop\OIDCの死骸.txt"

# ファイルを一行ずつ読み取る（ストリーム的な読み方）
with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip(),flush=True )  # 各行が届いたらすぐ処理
