from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langchain.vectorstores import Chroma
import gradio as gr

from src.model import load_model
from src.constants import (DEVICE_TYPE,
                        MODEL_ID, 
                        MODEL_BASENAME, 
                        PERSIST_DIRECTORY,
                        EMBEDDING_MODEL_NAME
                        )

# load digested docs
embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, 
                                           model_kwargs={"device": DEVICE_TYPE})

# load the vectorstore
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
    )
retriever = db.as_retriever()

# create prompt template and buffer memory
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
just say that you don't know, don't try to make up an answer.

{context}

{history}
Question: {question}
Helpful Answer:"""
prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
memory = ConversationBufferMemory(input_key="question", memory_key="history")

# load model
llm = load_model(DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

# build web ui
with gr.Blocks(gr.themes.Soft()) as demo:
    header = """
    <div align="left">
    <span style="display: inline-block; vertical-align: middle; font-size: 45px; font-weight:bold;"> Copilot </span>
    </div>
    """

    subheader = """
    ## 👋 Ask anything you want to know about RL! 
    """
    
    footer = """
    ## ⛰️ This copilot is built based on [LocalGPT](https://github.com/PromtEngineer/localGPT);

    ## 🦙 The base model is [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.3) of [LMSYS](https://lmsys.org/blog/2023-03-30-vicuna/);

    ## 🪪 [MIT License](https://opensource.org/license/mit/).
    """

    state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=5, min_width=100):
            blank1 = gr.Button(visible=False)

        with gr.Column(scale=90):
            with gr.Row():
                with gr.Column(scale=30):
                    with gr.Row():
                        gr.Image(value="images/logo_horizontal.svg", 
                                container=False, 
                                show_label=False, 
                                show_download_button=False,
                                show_share_button=False,
                                min_width=254
                                )
                        gr.HTML(header)
                with gr.Column(scale=70):
                    gr.Button(visible=False)

            gr.Markdown(subheader)
            chatbot = gr.Chatbot(avatar_images=["images/smile.png", "images/astronaut.png"],
                                 height=600
                                 )

            with gr.Row():
                with gr.Column(scale=8):
                    msg = gr.Textbox(show_label=False,
                                     placeholder="Enter text and press ENTER", container=False)
                with gr.Column(scale=2, min_width=60):
                    send_btn = gr.Button(value="Submit", variant="primary")
            
            with gr.Row():
                upvote_btn = gr.Button(value="👍  Upvote")
                downvote_btn = gr.Button(value="👎  Downvote")
                flag_btn = gr.Button(value="⚠️  Flag")
                regenerate_btn = gr.Button(value="🔄  Regenerate")
                clear_btn = gr.ClearButton([msg, chatbot], value="🗑️  Clear history")
            
            gr.Markdown(footer)
        
        with gr.Column(scale=5, min_width=100):
            blank2 = gr.Button(visible=False)

    def respond(message, chat_history):
        # answer = "I'm very hungry"
        # get the answer from the chain
        res = qa(message)
        answer, docs = res["result"], res["source_documents"]

        # print("----------------------------------SOURCE DOCUMENTS---------------------------")
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)
        # print("----------------------------------SOURCE DOCUMENTS---------------------------")

        chat_history.append((message, answer))
        return "", chat_history, answer
    
    def regenerate(state, chat_history):
        return state

    msg.submit(respond, [msg, chatbot], [msg, chatbot, state])

    send_btn.click(respond, [msg, chatbot], [msg, chatbot, state])

    # regenerate_btn.click(regenerate, [chatbot], [chatbot])

if __name__ == "__main__":
    demo.launch()

