import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model = "llama3-70b-8192"
)
st.title("Anime Story Generator")

topic = st.sidebar.selectbox("Genre", ("Action", "Adventure", "Horror", "Psychological", "Isekai", "Thriller", "Slice of life", "Murim", "Tragedy"))

def generate_title_char(topic):
    prompt_title = PromptTemplate(
        input_variables=["topic"],
        template="Generate a fancy and suitable title for the theme {topic}. The title should not contain more than 4 words. Limit the number of titles you are suggesting to one. Avoid these line at the start - Here's a fancy and suitable title for the theme Action:")
    title_chain = LLMChain(llm=llm, prompt=prompt_title, output_key="title")

    prompt_char_names=PromptTemplate(
        input_variables=["title"],
        template="Generate few character names suited for the title {title}. Avoid these lines at the start - Here are a few character name suggestions that fit the title. Give the names one in each line in numeric points format along with their role.")
    char_chain = LLMChain(llm=llm, prompt=prompt_char_names, output_key= "characters")

    prompt_story=PromptTemplate(
        input_variables=["topic", "title", "characters"],
        template="Generate a gripping anime storyline for the title '{title}'. Focus on themes of {topic} and include the characters {characters} in the story. Keep the stroy line brief with just 200 words.")
    story_chain = LLMChain(llm=llm, prompt=prompt_story, output_key= "story")

    seq_chain = SequentialChain(
        chains = [title_chain,char_chain, story_chain],
        input_variables = ["topic"],
        output_variables = ["title","characters", "story"])
    response = seq_chain({"topic" : topic})
    return response

if topic:
    response = generate_title_char(topic)
    st.header(response["title"])
    st.header("Characters")
    st.write(response["characters"])
    st.header("Basic Story Line")
    st.write(response["story"])
    