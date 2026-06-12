from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
import torch
import json
import os

# ══════════════════════════════════════════════════════════════════════
# 0. 全域設定
# ══════════════════════════════════════════════════════════════════════
MAX_SEQ_LENGTH = 2048
SEED = 3407

# ──────────────────── 外部 System Prompt（易於維護）────────────────────
with open("__history/SystemPrompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

# ══════════════════════════════════════════════════════════════════════
# 1. 載入模型（QLoRA）
# ══════════════════════════════════════════════════════════════════════
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/data/models/Qwen3-14B",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    # device_map={"": 0} 確保單 GPU 推論/訓練行為一致；
    # 若要多 GPU 訓練再改回 "auto"
    device_map={"": 0},
)

# ── LoRA 設定 ─────────────────────────────────────────────────────────
# r=16, alpha=16（alpha/r=1）：資料量約 400 筆，較小的 rank 搭配等比例 alpha
# 可降低過擬合風險；若驗證 loss 明顯高於訓練 loss，再考慮降到 r=8。
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,   # 防過擬合
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ══════════════════════════════════════════════════════════════════════
# 2. 載入資料
# ══════════════════════════════════════════════════════════════════════
with open("__history/train.jsonl", "r", encoding="utf-8") as f:
    train_raw = [json.loads(line) for line in f if line.strip()]

with open("__history/eval.jsonl", "r", encoding="utf-8") as f:
    eval_raw = [json.loads(line) for line in f if line.strip()]

train_dataset = Dataset.from_list(train_raw)
eval_dataset  = Dataset.from_list(eval_raw)

def format_data(example):
    """
    統一注入外部 SystemPrompt，並強制關閉 Qwen3 的 thinking 模式。

    【為何要關 enable_thinking】
    客服 SFT 資料沒有思考鏈，Qwen3 預設會在 assistant 回覆前
    插入 <think>...</think>。若訓練時不關閉，模型會學到輸出
    空的 <think></think>，推論行為會不一致。
    """
    messages = list(example["messages"])   # 避免 in-place 修改污染原資料

    # 統一替換或插入 system prompt
    if messages and messages[0]["role"] == "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages[1:]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  
    )
    return {"text": text}

# 先 map 格式化，再做 token 長度檢查
train_dataset = train_dataset.map(format_data, remove_columns=train_dataset.column_names)
eval_dataset  = eval_dataset.map(format_data,  remove_columns=eval_dataset.column_names)

# ── Token 長度健康檢查 ────────────────────────────────────────────────
lens = [len(tokenizer(t)["input_ids"]) for t in train_dataset["text"]]
p95  = sorted(lens)[int(len(lens) * 0.95)]
print(f"[長度統計] max={max(lens)}, p95={p95}, mean={sum(lens)//len(lens)}")
if max(lens) > MAX_SEQ_LENGTH:
    print(f"有樣本超過 MAX_SEQ_LENGTH({MAX_SEQ_LENGTH})，將被截斷，請考慮清理過長對話")

print(f"[資料筆數] 訓練：{len(train_dataset)}，驗證：{len(eval_dataset)}")

# ══════════════════════════════════════════════════════════════════════
# 3. 訓練設定
# ══════════════════════════════════════════════════════════════════════
# 步數估算（約 400 筆，train ≈ 360 筆）：
#   有效 batch = per_device(1) × grad_accum(8) = 8
#   每 epoch 約 360/8 = 45 步
#   6 epochs 總計約 270 步
#   eval_steps=15 → 每 1/3 epoch 評估一次，early_stopping_patience=3 → 45 步無改善停止

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    args=TrainingArguments(
        output_dir="./finetune_model/cetustek_qwen3_14b_qlora_v5",

        # ── Batch ──────────────────────────────────────────────────
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,      # 有效 batch=8，更新頻率充足

        # ── Epoch / Steps ─────────────────────────────────────────
        num_train_epochs=6,                 # 折中值；early stopping 會自動提早結束

        # ── Learning Rate ─────────────────────────────────────────
        learning_rate=2e-5,                 # 保守值
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,                   # 約 27 步 warmup，防梯度爆炸

        # ── Regularisation ────────────────────────────────────────
        weight_decay=0.01,                  # 防過擬合（搭配 lora_dropout）

        # ── Precision / Optimizer ─────────────────────────────────
        bf16=True,
        optim="paged_adamw_8bit",           # 比 adamw_8bit 更省視訊記憶體

        # ── Logging & Checkpointing ───────────────────────────────
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=15,
        save_strategy="steps",
        save_steps=15,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
        seed=SEED,
    ),
)

# ── Loss Masking：只計算 assistant 回覆的損失 ─────────────────────────
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ── Loss Mask 驗證：確認 mask 有正確套用 ──────────────────────────────
_ex   = trainer.train_dataset[0]
_kept = [t for t, l in zip(_ex["input_ids"], _ex["labels"]) if l != -100]
print("=" * 50)
print("【Loss Mask 驗證】以下應只包含 assistant 回覆內容：")
print(tokenizer.decode(_kept))
print("=" * 50)

# ══════════════════════════════════════════════════════════════════════
# 4. 開始訓練
# ══════════════════════════════════════════════════════════════════════
trainer.train()

# ══════════════════════════════════════════════════════════════════════
# 5. 儲存最佳模型(loss 最低)
# ══════════════════════════════════════════════════════════════════════
model.save_pretrained("./finetune_model/cetustek_qwen3_14b_qlora_v5")
tokenizer.save_pretrained("./finetune_model/cetustek_qwen3_14b_qlora_v5")
print("訓練完成，最佳 LoRA adapter 已儲存至 ./finetune_model/cetustek_qwen3_14b_qlora_v5")

# ══════════════════════════════════════════════════════════════════════
# 6. 驗證集推論 + bge-m3 相似度評估
# ══════════════════════════════════════════════════════════════════════
print("\n開始進行驗證集相似度評估...")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from rouge_score import rouge_scorer as rouge_scorer_module
    import numpy as np
except ImportError:
    print("請先安裝套件：pip install sentence-transformers scikit-learn rouge-score numpy")
    exit()

# 切回推論模式（Unsloth 加速）
FastLanguageModel.for_inference(model)

with open("eval.jsonl", "r", encoding="utf-8") as f:
    eval_for_infer = [json.loads(line) for line in f if line.strip()]

ground_truths = []
predictions   = []

print(f"評估筆數：{len(eval_for_infer)}")

for idx, data in enumerate(eval_for_infer):
    try:
        messages = data["messages"]

        # 確保最後一句是 assistant，倒數第二句是 user
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            continue

        ground_truth = messages[-1]["content"]
        prompt_msgs  = list(messages[:-1])

        # 注入 System Prompt（與訓練格式完全一致）
        if prompt_msgs and prompt_msgs[0]["role"] == "system":
            prompt_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + prompt_msgs[1:]
        else:
            prompt_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + prompt_msgs

        prompt = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # 與訓練格式一致
        )

        inputs  = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            do_sample=False,         # greedy decode，評估結果最穩定
        )

        pred_text = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        ground_truths.append(ground_truth)
        predictions.append(pred_text)

        if idx < 3:
            print(f"\n[樣本 {idx+1}]")
            print(f"Ground Truth : {ground_truth}")
            print(f"預測         : {pred_text}")

    except Exception as e:
        print(f"[跳過第 {idx} 筆] 錯誤：{e}")
        continue

# ── Ground Truth 長度分布 ─────────────────────────────────────────────
gt_lens  = [len(tokenizer(gt)["input_ids"]) for gt in ground_truths]
gt_lens_sorted = sorted(gt_lens)
print("\n" + "=" * 55)
print("【Ground Truth 長度分布】")
print(f"  max  : {max(gt_lens)}")
print(f"  p95  : {gt_lens_sorted[int(len(gt_lens) * 0.95)]}")
print(f"  p50  : {gt_lens_sorted[int(len(gt_lens) * 0.50)]}")
print(f"  mean : {sum(gt_lens) // len(gt_lens)}")
print(f"  min  : {min(gt_lens)}")
print("建議 max_new_tokens 設為 p95 的 1.2 倍左右")
print("=" * 55)

# ── bge-m3 Cosine Similarity ──────────────────────────────────────────
print("\n載入 bge-m3 模型進行向量相似度評估...")
embedder        = SentenceTransformer("BAAI/bge-m3")
gt_embeddings   = embedder.encode(ground_truths, show_progress_bar=True)
pred_embeddings = embedder.encode(predictions,   show_progress_bar=True)

cos_scores = [
    cosine_similarity(
        gt_embeddings[i].reshape(1, -1),
        pred_embeddings[i].reshape(1, -1)
    )[0][0]
    for i in range(len(ground_truths))
]

# ── ROUGE-L ───────────────────────────────────────────────────────────
scorer       = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=False)
rouge_scores = [
    scorer.score(gt, pred)["rougeL"].fmeasure
    for gt, pred in zip(ground_truths, predictions)
]

print("\n" + "=" * 55)
print(f"評估筆數              : {len(ground_truths)}")
print(f"bge-m3 Cosine 平均   : {np.mean(cos_scores):.4f} / 1.0000")
print(f"ROUGE-L 平均          : {np.mean(rouge_scores):.4f} / 1.0000")
print("=" * 55)