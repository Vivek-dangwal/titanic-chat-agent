import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv
import os

# 1. Load the Secret Key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 2. Advanced Styling (Cruise Background & Labels)
st.set_page_config(page_title="Titanic AI Agent", page_icon="🚢", layout="wide")

# This adds a cool cruise ship background and styles the text
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
        url("https://images.unsplash.com/photo-1548574505-5e239809ee19?q=80&w=2000");
        background-size: cover;
        color: white;
    }
    .main-title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
    }
    .dev-name {
        font-size: 20px;
        text-align: center;
        color: #FFD700;
        margin-bottom: 30px;
    }
    .stTextInput label, .stCheckbox label {
        color: white !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">🚢 Titanic Dataset Chat Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="dev-name">Developer: Vivek Dangwal</p>', unsafe_allow_html=True)

# 3. Load Data
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

try:
    df = load_data()
    
    # 4. Initialize Brain (Modern Tool-Calling Logic)
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=api_key
    )

    # 5. Create Agent with specific instructions to avoid errors
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        agent_type="tool-calling" # This is the most stable version
    )

    # 6. Chat Interface
    with st.container():
        user_question = st.text_input("What would you like to know about the Titanic passengers?")

    if user_question:
        with st.spinner("🚢 Diving into the data..."):
            try:
                # Force the robot to handle charts correctly
                response = agent.invoke(user_question)
                st.success("Analysis Complete")
                st.write(response["output"])
                
                # Show charts if generated
                if plt.get_fignums():
                    st.pyplot(plt.gcf())
                    plt.clf()
                    
            except Exception as e:
                st.error(f"Error: {e}")

    # 7. Raw Data Peek
    if st.checkbox("Show Sample Passenger List"):
        st.dataframe(df.head(10))

except FileNotFoundError:
    st.error("Missing 'titanic.csv'! Please place it in the project folder.")