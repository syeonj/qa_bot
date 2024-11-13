import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables
load_dotenv()

# Streamlit app settings
st.set_page_config(
    page_title="직터뷰",
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# OpenAI API 설정 (필요시)
# openai.api_key = os.environ.get("OPENAI_API_KEY")

# 대화 기록 초기화
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# QA Chain 설정 (한 번만 학습)
if "qa_chain" not in st.session_state:
    # YouTube 동영상 URL 및 오디오 저장 경로
    urls = [
      'https://www.youtube.com/watch?v=YxYWhn8gpGA&list=PLC9G30QJs92gI1LQUPJmYrv0sQzIIcGV8&index=9'
    ]

    save_dir = "./youtube_audios/"

    # 동영상을 텍스트로 변환
    from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import OpenAIWhisperParser
    from functools import reduce
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings

    loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
    docs = loader.load()

    # Define word pairs to replace
    replace_pairs = {
        '제정팀': '재정팀', '아리젤략실':'RE전략실', '신구사업': '신규사업', '기관한': '기반한', '페르소란' : '페르소나', '라팡':'라이프파크', 
        'ASI': '계리사 1차', '부조용실':'부조정실', '다벤치':'다빈치','인사트':'인사이트',
        '안보험': '암보험','중심파트': '종신파트','중심보험': '종신보험','지하의보험': 'GI보험','개리지원팀': '계리지원팀','선임개리파트': '선임계리파트',
'개리': '계리',
'보험개리사': '보험계리사',
'개리직무': '계리직무',
'국어 공문학적': '국어 국문학적',
'두갈식': '두괄식',
'하나 생명': '한화생명',
'하나 이글스': '한화이글스',
'생부업계': '생보업계',
'이데이': '이대희'
}

    # Replace words in all docs
    for doc in docs:
        if hasattr(doc, 'page_content'):
            doc.page_content = (
                doc.page_content
                if not any(old in doc.page_content for old in replace_pairs)
                else reduce(lambda text, pair: text.replace(pair[0], pair[1]), replace_pairs.items(), doc.page_content)
            )

    # 텍스트 전처리 및 벡터화
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(splits, embeddings)

    # 콜백 핸들러 클래스 정의
    class StreamCallback(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs):
            st.text(f"{token}")

    st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            streaming=True,
            callbacks=[StreamCallback()],
        ),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )

# Streamlit Query UI
st.title("직터뷰 🧑‍⚕️")
st.text("궁금하신 정보에 대해 물어보세요 🔎 ")

# 대화 기록 표시
if st.session_state["chat_answer_history"]:
    for index, (query, answer) in enumerate(zip(st.session_state["user_prompt_history"], st.session_state["chat_answer_history"])):
        widget_key1 = f"query_{index}"
        widget_key2 = f"ans_{index}"
        message(query, is_user=True, key=widget_key1)
        message(answer, key=widget_key2)

# 사용자 입력과 제출 버튼을 한 줄로 배치
with st.form("form"):
    cols = st.columns([8, 1])  # 첫 번째 컬럼은 입력란, 두 번째는 버튼
    user_input = cols[0].text_input("궁금하신 내용을 입력하세요.", key="user_input", label_visibility="collapsed")
    submit = cols[1].form_submit_button("물어보기")

# 사용자 입력을 처리하여 QA 응답 생성
if user_input and submit:
    # 스피너를 프롬프트 위에 표시
    with st.spinner("답변 생성중..."):
        response = st.session_state["qa_chain"].run(user_input)

        # 대화 기록 저장
        st.session_state["user_prompt_history"].append(user_input)
        st.session_state["chat_answer_history"].append(response)
        st.session_state["chat_history"].append((user_input, response))

    # 새로운 메시지가 추가된 후 대화 기록 갱신
    st.experimental_rerun()

# # requirements.txt 생성
# def generate_requirements():
#     requirements = [
#         "streamlit",
#         "python-dotenv",
#         "streamlit-chat",
#         "langchain-openai",
#         "langchain-community",
#         "faiss-cpu",
#     ]
#     with open("requirements.txt", "w") as f:
#         f.write("\n".join(requirements))

# generate_requirements()

# # packages.txt 생성
# def generate_packages():
#     packages = [
#         "ffmpeg",  # For handling YouTube audio files
#     ]
#     with open("packages.txt", "w") as f:
#         f.write("\n".join(packages))

# generate_packages()
