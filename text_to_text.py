from openai import OpenAI

# OpenAI APIキーを設定
api_key = ""

from openai import OpenAI

client = OpenAI(api_key=api_key)

user_input = "下北でお勧めグルメ教えて"

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": user_input}],
    max_tokens=200,
    temperature=0.5,
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)

print()
