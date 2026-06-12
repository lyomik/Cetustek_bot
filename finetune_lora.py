from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
import torch
import json
import os

# ── 設定 ────────────────────────────────────────────────────────────────
# 把序列長度集中成一個變數，避免「載入用 2048、訓練用 1024」這種不一致導致截斷。
# 多輪對話 + 外部 SystemPrompt 很容易超過 1024，被截斷會把結尾的 assistant 回覆切掉，
# 那一筆就等於白訓練（loss mask 找不到 response）。下面第 2 步會實際量長度再決定。
MAX_SEQ_LENGTH = 2048
V = "v4"

# ── 0. 讀取外部 System Prompt ────────────────────────────────────────────
with open("SystemPrompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# ── 1. 載入模型 ──────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/data/models/Qwen3-14B",
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
    dtype = torch.bfloat16,
    device_map = {"": 0},
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0.05, # lora_dropout:0.05，隨機丟棄 5% 的神經元連接，強制模型不要死背特定字句，防止 Overfitting
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)


# ── 2. 載入資料集 ─────────────────────────────────────────────────────────
with open("train.jsonl", "r", encoding="utf-8") as f:
    train_raw = [json.loads(line) for line in f if line.strip()]

with open("eval.jsonl", "r", encoding="utf-8") as f:
    eval_raw = [json.loads(line) for line in f if line.strip()]

train_dataset = Dataset.from_list(train_raw)
eval_dataset = Dataset.from_list(eval_raw)


def format_data(example):
    # 不在原 list 上 in-place 修改，避免 map 重跑時污染原資料
    messages = list(example["messages"])
    # 如果第一條不是 system，就插入；否則覆蓋成統一的 SYSTEM_PROMPT
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages[1:]

    # 【關鍵】客服資料沒有思考鏈，Qwen3 預設會開 thinking 並塞入 <think> 區塊，
    # 必須關掉並保持「訓練 / 推論」一致，否則模型會學到輸出空的 <think></think>。
    return { "text" : tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = False,
        enable_thinking = False,
    )}

# 套用格式
train_dataset = train_dataset.map(format_data)
eval_dataset = eval_dataset.map(format_data)

# 【建議檢查】量實際 token 長度，確認 MAX_SEQ_LENGTH 蓋得過 p95，避免截斷
lens = [len(tokenizer(t)["input_ids"]) for t in train_dataset["text"]]
print(f"token 長度：max={max(lens)}, p95={sorted(lens)[int(len(lens)*0.95)]}, mean={sum(lens)//len(lens)}")
if max(lens) > MAX_SEQ_LENGTH:
    print(f"⚠️ 有樣本超過 MAX_SEQ_LENGTH({MAX_SEQ_LENGTH})，會被截斷，請考慮調高或清理過長對話")

print(f"訓練筆數：{len(train_dataset)}，驗證筆數：{len(eval_dataset)}")


# ── 3. 訓練 ──────────────────────────────────────────────────────────────
# 資料 423 筆，batch_size=1 * 8 = 8。每跑完 1 個 Epoch 需要 423/8 ≈ 53 步。
# 4 個 Epoch 總共約 212 步。
# 設定 eval_steps=15 (大約每跑 1/3 個 Epoch 就考一次試)，監控頻率最完美。

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,          
    dataset_text_field = "text",          
    max_seq_length = MAX_SEQ_LENGTH,      
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,     
        warmup_ratio = 0.05,                 # 👉 從 0.03 微調到 0.05，讓前 10 步慢慢學習，防梯度爆炸。
        num_train_epochs = 4,                
        learning_rate = 1e-4,                
        weight_decay = 0.01,                 # 👉 新增權重衰減，搭配 dropout 組成防過擬合雙重護盾。
        bf16 = True,
        logging_steps = 5,
        eval_strategy = "steps",
        eval_steps = 15,                     # 👉 配合 batch 數量，改為每 15 步評估一次
        save_strategy = "steps",
        save_steps = 15,                     # 👉 配合 eval_steps 同步儲存
        save_total_limit = 2,
        load_best_model_at_end = True,       
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        output_dir = "./finetune_model/cetustek_qwen3_14b_output_v4",
        optim = "adamw_8bit",
        lr_scheduler_type = "cosine",
        report_to = "none",
    ),
    # eval_loss 連續 3 次沒進步就早停（45步內沒進步就停損）
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)],
)

# Loss Masking (只計算 Assistant 輸出的損失)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

_ex = trainer.train_dataset[0]
_kept = [t for t, l in zip(_ex["input_ids"], _ex["labels"]) if l != -100]
print("=" * 40)
print("【Loss Mask 驗證】以下應只包含 assistant 回覆：")
print(tokenizer.decode(_kept))
print("=" * 40)

trainer.train()

# ── 4. 儲存 ──────────────────────────────────────────────────────────────
# 因 load_best_model_at_end=True，這裡存的是 eval_loss 最低的版本（LoRA adapter）
model.save_pretrained(f"./finetune_model/cetustek_qwen3_14b_lora_{V}")
tokenizer.save_pretrained(f"./finetune_model/cetustek_qwen3_14b_lora_{V}")
print(f"🎉 訓練完成，最佳模型已儲存至 ./finetune_model/cetustek_qwen3_14b_lora_{V}")

# ==========================================
# 5. 模型驗證與特徵向量相似度比對 (bge-m3)
# ==========================================
print("\n🚀 訓練完畢！開始進行驗證集 (bge-m3) 相似度比對測試...")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import json
except ImportError:
    print("❌ 請先安裝套件: pip install sentence-transformers scikit-learn numpy")
    exit()

# 開啟 Unsloth 的推論加速模式 (速度快 2 倍)
FastLanguageModel.for_inference(model)

# 讀取剛剛切出來的 eval.jsonl (驗證集)
with open("eval.jsonl", "r", encoding="utf-8") as f:
    eval_data = [json.loads(line) for line in f if line.strip()]

ground_truths = []
predictions = []

print(f"總共要評估 {len(eval_data)} 筆對話，正在讓模型作答中 (請稍候)...")

for idx, data in enumerate(eval_data):
    messages = data["messages"]
    
    # 確保對話最後一句是 assistant，倒數第二句是 user
    if messages[-1]["role"] == "assistant" and messages[-2]["role"] == "user":
        # 標準答案 (Ground Truth)
        ground_truth = messages[-1]["content"]
        
        # 截斷對話：只保留到 user 問完，把最後一句 assistant 刪掉，讓模型自己產生
        prompt_messages = messages[:-1]
        
        # 使用 tokenizer 將對話轉為模型看得懂的 prompt 格式
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # 讓模型生成回答 (設定 temperature=0.1 確保回答穩定不亂飄)
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, temperature=0.1, top_p=0.1)
        
        # 解碼出模型生成的純文字回答
        pred_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        ground_truths.append(ground_truth)
        predictions.append(pred_text.strip())
        
        # 印出前 3 筆，讓你用肉眼感受一下模型的作答狀況
        if idx < 3:
            print(f"\n[測試 {idx+1}]")
            print(f"🎯 真實 SOP : {ground_truth}")
            print(f"🤖 模型預測 : {pred_text.strip()}")

# ── 進入 bge-m3 評分階段 ──
print("\n⏳ 正在載入 BAAI/bge-m3 模型進行向量比對 (初次執行會下載約 2.2GB 模型)...")
embedder = SentenceTransformer('BAAI/bge-m3')

print("📐 正在計算餘弦相似度 (Cosine Similarity)...")
# 將文字轉為向量
gt_embeddings = embedder.encode(ground_truths)
pred_embeddings = embedder.encode(predictions)

scores = []
for i in range(len(ground_truths)):
    # 逐筆比對矩陣相似度
    sim_score = cosine_similarity(gt_embeddings[i].reshape(1, -1), pred_embeddings[i].reshape(1, -1))[0][0]
    scores.append(sim_score)

# 算出整體平均分數
avg_score = np.mean(scores)

print("\n" + "="*50)
print(f"🏆 模型驗證集整體平均相似度得分 (bge-m3): {avg_score:.4f} / 1.0000")
print("="*50)

# # (選擇性) 將詳細的評分結果存成 CSV 留存研究數據
# import pandas as pd
# results_df = pd.DataFrame({
#     "Ground Truth": ground_truths,
#     "Model Prediction": predictions,
#     "Similarity Score": scores
# })
# results_df.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")
# print("📁 詳細評分結果已儲存為 evaluation_results.csv")