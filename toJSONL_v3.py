import pandas as pd
import json
import glob
import random

# ==========================================
# 1. 抓取資料夾內的 Excel (.xlsx) 檔案
# ==========================================

excel_files = glob.glob("*.xlsx")
dataset = []
system_prompt = "你現在是「鯨躍科技」的專業客服人員，鯨躍科技是電子發票加值中心。"

# ==========================================
# 2. 讀取與轉換邏輯 (支援多個 Excel 檔與多分頁)
# ==========================================

for file in excel_files:
    print(f"👀 正在讀取 Excel 檔案：{file}")
    try:
        # 讀取 Excel 內的所有工作表 (sheet)
        all_sheets = pd.read_excel(file, sheet_name=None)
        
        for sheet_name, df in all_sheets.items():
            print(f"  - 處理分頁 [{sheet_name}]...")
            for index, row in df.iterrows():
                current_messages = [
                    {"role": "system", "content": system_prompt}
                ]
                
                for col_name in df.columns:
                    cell_value = str(row[col_name]).strip()
                    
                    if not cell_value or cell_value.lower() in ['nan', 'nat', 'none', '-']:
                        continue
                    
                    col_name_lower = str(col_name).lower()
                    if 'assistant' in col_name_lower or '客服' in col_name_lower or '鯨躍' in col_name_lower:
                        role = "assistant"
                    elif 'user' in col_name_lower or '客戶' in col_name_lower:
                        role = "user"
                    else:
                        continue 
                        
                    current_messages.append({"role": role, "content": cell_value})
                
                if len(current_messages) > 1:
                    dataset.append({"messages": current_messages})

    except Exception as e:
        print(f"讀取檔案 {file} 失敗: {e}")
        continue

# ==========================================
# 3. 打亂並切割成「訓練集(Train) 85%」與「驗證集(Eval) 15%」
# ==========================================

random.seed(42)
random.shuffle(dataset)

split_index = int(len(dataset) * 0.85)
train_dataset = dataset[:split_index]
eval_dataset = dataset[split_index:]

def save_jsonl(data_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

save_jsonl(train_dataset, "train.jsonl")
save_jsonl(eval_dataset, "eval.jsonl")

print(f"\n✅ 轉換與切割大成功！(切割比例 85/15)")
print(f"📊 總資料筆數：{len(dataset)} 筆")
print(f"🚀 訓練集 (Train) 產出：{len(train_dataset)} 筆 -> 已儲存為 train.jsonl")
print(f"🔬 驗證集 (Eval) 產出：{len(eval_dataset)} 筆 -> 已儲存為 eval.jsonl")