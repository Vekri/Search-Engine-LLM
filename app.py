import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool
from dotenv import load_dotenv

# Load environment variables if needed
load_dotenv()

# Safe DuckDuckGo Search wrapper
def safe_duckduckgo_search(query: str) -> str:
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

# Tool setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = Tool(
    name="Search",
    func=safe_duckduckgo_search,
    description="Web search using DuckDuckGo"
)

# Streamlit UI
st.title("🔎 LangChain - Chat with Search")

"""
Ask questions about India, AI, science, or anything else. The app uses Wikipedia, Arxiv, and web search tools via LangChain.
"""

# Sidebar for API Key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input with NEW placeholder
if prompt := st.chat_input("Ask me anything about India, AI, science, or more..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        except Exception as e:
            response = f"An error occurred while running the agent: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
