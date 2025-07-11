# main.py

from model import load_model

# ✅ 프롬프트 생성 함수
def build_prompt(user_input):
    return f"""당신은 친절한 ETL 전문가 AI입니다.
사용자와 대화를 나누는 중입니다.

👤 사용자: {user_input}
🤖 AI:"""

# ✅ 실행 로직
def main():
    tokenizer, model = load_model()

    print("🤖 ETL 전문가 AI와 대화해보세요! 종료하려면 'exit'을 입력하세요.")

    while True:
        user_input = input("👤 사용자: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("👋 대화를 종료합니다.")
            break

        prompt = build_prompt(user_input)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )

        # 🤖 AI 응답 추출
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # response_lines = response.split("🤖 AI:")
        answer = response.strip()

        print(f"🧠 AI 응답: {answer}\n")
        # break

if __name__ == "__main__":
    main()