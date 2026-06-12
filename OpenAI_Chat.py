# ======= 注意：使用前開啟虛擬環境 WebUI_venv =======
import gradio as gr
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ====== 設定 ======
API_URL    = "http://localhost:8002/v1"
MODEL      = "cetustek-bot"
GREETING   = "您好，這裡是 Cetustek！ 有什麼可以為您協助？"
CHROMA_DIR = "./chroma_sop"   # 改成你實際的路徑
# ==================

client = OpenAI(base_url=API_URL, api_key="not-needed")

# ==== 載入現有向量庫 ====
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)
vector_db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)
print(f"✅ 向量庫已載入，共 {vector_db._collection.count()} 筆段落")


def chat_stream(user_input, history, custom_system, temp, top_p):
    system = custom_system

    results = vector_db.similarity_search_with_relevance_scores(user_input, k=3)
    relevant = [doc.page_content for doc, score in results if score > 0.5]
    if relevant:
        context = "\n---\n".join(relevant)
        system += f"\n\n以下是相關 SOP 參考資料，請依據此回覆：\n{context}"

    messages = [{"role": "system", "content": system}]
    for msg in history:
        if msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temp,
            top_p=top_p,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=True
        )
        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                partial_message += chunk.choices[0].delta.content
                yield partial_message

    except Exception as e:
        yield f"【錯誤】無法連接到 vLLM 伺服器：{str(e)}"


# ====== 前端介面 ======
with gr.Blocks(title="Cetustek AI 助手") as demo:
    gr.Markdown("# 🐳 Cetustek 鯨躍科技 AI 助手")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            with gr.Accordion("⚙️ 模型進階參數", open=True):
                system_prompt = gr.Textbox(
                    value="您是一位專業的 AI 助理。請提供詳細完整的回答，不要省略重要資訊。你僅能提供客服回覆，無法幫客戶操作",
                    label="System Prompt",
                    lines=3
                )
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Temperature")
                top_p       = gr.Slider(0.0, 1.0, value=0.8, step=0.1, label="Top P")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": GREETING}],
                height=600,
                show_label=False
            )
            gr.ChatInterface(
                fn=chat_stream,
                chatbot=chatbot,
                additional_inputs=[system_prompt, temperature, top_p],
                textbox=gr.Textbox(placeholder="請輸入您的問題...", container=False, scale=7)
            )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8001,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue")
    )