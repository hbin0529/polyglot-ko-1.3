
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    model_name = "EleutherAI/polyglot-ko-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto"  # 또는 "float16" (GPU 사용시)
    )
    return tokenizer, model