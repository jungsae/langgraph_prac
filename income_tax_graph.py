# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings

embedding_function = UpstageEmbeddings(
    model="embedding-query"
)

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name="income_tax_collection",
    persist_directory="./income_tax_collection"
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

graph_builder = StateGraph(AgentState)

# %%
def retrieve(state: AgentState):
    query = state["query"]
    docs = retriever.invoke(query)
    return {"context": docs}

# %%
# # LOCAL LLM(Qwen3-4B-Instruct)
# from langchain_openai import ChatOpenAI

# PORT = 8000

# LOCAL_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# llm = ChatOpenAI(
#     model=LOCAL_MODEL,
#     base_url=f"http://localhost:{PORT}/v1",
#     api_key="NOT_USED",
#     temperature=0.05,
#     top_p=0.95,
#     extra_body={
#         "top_k": 60,
#     },
#     max_tokens=1024,
# )

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=900,
    temperature=0
)

# # llm = ChatOpenAI(
# #     model="gpt-5-mini",
# #    reasoning_effort="low",
# #     model_kwargs={           
# #         "verbosity": "low"
# #     },
# # )

# %%
from langchain import hub

generate_prompt = hub.pull("rlm/rag-prompt")

def generate(state: AgentState):
    context = state["context"]
    query = state["query"]
    rag_chain = generate_prompt | llm
    response = rag_chain.invoke({"question": query, "context": context})
    return {"answer": response.content}

# %%
from typing import Literal
from langchain import hub


doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    query = state["query"]
    context = state["context"]
    print(f'context: {context}')
    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({"question": query, "documents": context})
    print(f'doc_relevance_response: {response}')
    if response['Score'] == 1:
        return 'relevant'
        
    return 'irrelevant'

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 => 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, dictionary를 참고해서 사용자의 질문을 변경해주세요
dictionary: {dictionary}
질문: {{query}}
""")

def rewrite(state: AgentState):
    query = state["query"]
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({"query": query})
    return {"query": response}

# %%
from langchain_core.prompts import PromptTemplate

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evatluating whether the student's answer is based on the documents or not.
Given documents, which are excerpts from a income tax law, and a student's answer;
If the studnets answer is based on documents, respond with "not hallucinated".
If the studnets answer is not based on documents, respond with "hallucinated".
So, respond with only "not hallucinated" or "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")

def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    query = state["answer"]
    context = state["context"]
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | llm | StrOutputParser()
    response = hallucination_chain.invoke({"student_answer": query, "documents": context})
    print(f'hallucination response: {response}')
    return response

# %%
from langchain import hub
helpfulness_prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state: AgentState):
    query = state["query"]
    answer = state["answer"]
    helpfulness_chain = helpfulness_prompt | llm
    response = helpfulness_chain.invoke({"question": query, "student_answer": answer})
    print(f'helpfulness response: {response}')
    if response['Score'] == 1:
        return 'helpful'
    return 'unhelpful'

def check_helpfulness(state: AgentState):
    return state

# %%
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("check_helpfulness", check_helpfulness)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges(
    "retrieve",
    check_doc_relevance,
    {
        "relevant": "generate",
        "irrelevant": END
    }
)

graph_builder.add_conditional_edges(
    "generate",
    check_hallucination,
    {
        "hallucinated": "generate",
        "not hallucinated": "check_helpfulness"
    }
)

graph_builder.add_conditional_edges(
    "check_helpfulness",
    check_helpfulness_grader,
    {
        "helpful": END,
        "unhelpful": "rewrite"
    }
)

graph_builder.add_edge("rewrite", "retrieve")

# %%
graph = graph_builder.compile()