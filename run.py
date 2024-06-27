from pathlib import Path
import streamlit as st
from ast import literal_eval

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableConfig
from sqlalchemy import create_engine
import sqlite3

from callbacks.capturing_callback_handler import playback_callbacks, CapturingCallbackHandler
from clear_results import with_clear_container

# custom
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

import json
import requests
import pandas as pd
import os
from uuid import uuid4
from typing import Any
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
import os
import time

from langchain.tools import tool

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults


DB_PATH = (Path(__file__).parent / "Chinook.db").absolute()
CHATBOT_URLS = {
    "chatbot1": "https://container-rag-challenge.ambitiousbeach-b93588ea.eastus.azurecontainerapps.io/chatbot1",
    "chatbot2": "https://container-rag-challenge.ambitiousbeach-b93588ea.eastus.azurecontainerapps.io/chatbot2",
}

files = list((Path(__file__).parent / "runs").glob("*"))
SAVED_SESSIONS = {}
for file in files:
    full_name = file.name
    if "eval_df" in full_name:
        continue
    name = full_name.split("_url_")[0]
    # SAVED_SESSIONS[name] = full_name
    SAVED_SESSIONS[full_name] = name

# SAVED_SESSIONS = {
    # "Younger generation (think gen Z) | Searching for advice on how to run a startup": "gen_z.pickle",
# }

st.set_page_config(
    page_title="Chatbot Agentic Eval", page_icon="🥷", layout="wide", initial_sidebar_state="collapsed"
)

"# 🥷 Chatbot Agentic Eval"

# Setup credentials in Streamlit
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to run your own custom questions."
)

TAVILY_API_KEY = st.sidebar.text_input(
    "Tavily API Key", type="password", help="Set this to run your own custom questions."
)

PINECONE_API_KEY = st.sidebar.text_input(
    "Pinecone API Key", type="password", help="Set this to run your own custom questions."
)

HF_KEY = st.sidebar.text_input(
    "Hugging Face API Key", type="password", help="Set this to run your own custom questions."
)

chatbot_api_key = st.sidebar.text_input(
    "Chatbot API Key", type="password", help="Set this to run your own custom questions."
)

agent_model = st.sidebar.selectbox(
    "Agent Model",
    [
        "gpt-3.5-turbo-0125",
        "gpt-4o",
    ],
    index=0,
)

chatbot_selection = st.sidebar.selectbox(
    "Chatbot URL",
    [
        "chatbot1",
        "chatbot2",
    ],
    index=0,
)

chatbot_url = CHATBOT_URLS[chatbot_selection]


if not TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = "something"
else:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    
if not PINECONE_API_KEY:
    PINECONE_API_KEY = "something"

if not HF_KEY:
    HF_KEY = "something"
    
if not chatbot_api_key:
    chatbot_api_key = "something"


profile_tools = [TavilySearchResults(max_results=1)]

import requests

if user_openai_api_key:
    openai_api_key = user_openai_api_key
    enable_custom = True
else:
    openai_api_key = "not_supplied"
    enable_custom = False

# Tools setup
profile_llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    openai_api_key=openai_api_key
)

intermediate_llm = ChatOpenAI(
    model=agent_model,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    openai_api_key=openai_api_key
)

from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)

model_name = 'text-embedding-3-small'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai_api_key,
    # chunk_size=1024
)

# spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'chatbot-knowledge-articles'

text_field = "text"

# switch back to normal index for langchain
if PINECONE_API_KEY != "something":
    index = pc.Index(index_name)

    from langchain.vectorstores import Pinecone

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )
else:
    vectorstore = None


# llm = OpenAI(temperature=0, openai_api_key=openai_api_key, streaming=True)

# Make the DB connection read-only to reduce risk of injection attacks
# See: https://python.langchain.com/docs/security # ?mode=ro
creator = lambda: sqlite3.connect(f"file:{DB_PATH}", uri=True)
db = SQLDatabase(create_engine("sqlite:///", creator=creator))

eval_df = pd.DataFrame(columns=["reforumlated_message", "response", "message_type", "is_no_answer", "is_helpful_answer", "irrelevant_answer", "non_actionable_user_like_response", "missing_claims_in_knowledgebase_rate", "missing_relevant_claims_in_system_response_rate", "personality_score", "avg. response length"])

# helper functions:
# ----------------------------------------------------
def get_relevant_article_snippets(query: str, k: int = 3):

    relevant_docs = vectorstore.similarity_search(
        query,  # our search query
        k=k
    )
    return relevant_docs

def get_response_personality(response: str):
    payload = {"inputs": response}

    API_URL = "https://api-inference.huggingface.co/models/Nasserelsaman/microsoft-finetuned-personality"
    headers = {"Authorization": f"Bearer {HF_KEY}"}

    response = requests.post(API_URL, headers=headers, json=payload)
    labels = response.json()
    if "estimated_time" in labels:
        time.sleep(labels["estimated_time"])
        response = requests.post(API_URL, headers=headers, json=payload)
        labels = response.json()

    try:
        labels = labels[0]
    except Exception as e:
        print(f"Ran into issue while fetching personality: {e}\n\n{labels}")
        return {"Agreeableness": 0, "Conscientiousness": 0, "Extroversion": 0, "Neuroticism": 0, "Openness": 0}
    
    return {
        "Agreeableness": [float(l['score']) for l in labels if l['label'] == "LABEL_0"][0],
        "Conscientiousness": [float(l['score']) for l in labels if l['label'] == "LABEL_1"][0],
        "Extroversion": [float(l['score']) for l in labels if l['label'] == "LABEL_2"][0],
        "Neuroticism": [float(l['score']) for l in labels if l['label'] == "LABEL_3"][0],
        "Openness": [float(l['score']) for l in labels if l['label'] == "LABEL_4"][0],
    }

# ----------------------------------------------------

# Step 1: Get User Profile
@tool
def GetUserProfile(initial_user_profile: str) -> str:
    """This should be the first function that runs. The inputted user profile needs to be expanded so that we can reliably use this for simulating the user interactions with the chatbot.
       This function is used to elaborate on the user profile which was provided.
       Params:
         - basic_user_info (str): The user characteristic that was provided to the system, which needs to be expanded. Use the full text provided as input. The system will further take care of it.
        
       Note: Use this only once per user.
    """
    profile_agent = create_react_agent(profile_llm, profile_tools, hub.pull("hwchase17/react"))
    profile_agent_executor = AgentExecutor(agent=profile_agent, tools=profile_tools, verbose=True, handle_parsing_errors=True)
    
    prompt = f"""I'm working on evaluating a chatbot system and I would like to draw-up a user profile based on a few initial characteristics I have about the user.
        This user would be interacting with the chatbot system and the user profile you generate would be helpful for another system to simulate the user interactions with the chatbot.
        The initial user characteristics are:
        {initial_user_profile}.
        
        Repeating the instructions:
        1. Your goal is only to Elaborate on the User Profile that is required.
        2. Write a single paragraph that describes the user including his name, age, personality, background, and intention for why he is here.
        3. Use the search api to get more information when necessary.
        4. The generated profile should start with 'user profile:' and then describe their full profile that will be passed to the next system."""
    user_profile = profile_agent_executor.invoke(
        {
            "input": prompt
        }
    )
    return user_profile["output"]

# Step 2: Generate Test Cases (This is ideally supposed to be query planning)
@tool
def TestCaseGenerator(existing_message_types: list[str], new_message_type: str) -> str:
    """You are an intelligent system that is stress testing a Chatbot system.
    You have previously generated some queries under some test categories that would strees test the system.
    Generate More Test case types, if you haven't completed all of listed ones. Else, return empty string.
    
    Params:
    - existing_message_types (list[str]): Collect all the existing message_types that you have already tested.
    - new_message_type (str): New message type for stress testing the LLM. You should use the test cases provided below. If you have completed all, return empty string.
    
    Do not retry same tests.
    
    Test cases:
    1. A random irrelevant OOD question
    2. relevant question using knowledgebase
    3. short follow-up question: s.t. it is ambigious w/o previous message.
    4. asking an actionable query that is not possible for a chatbot to do.
    5. social message like greeting the bot.
    6. requesting some sensitive company information.
    """
    if new_message_type in existing_message_types:
        return ""
    return new_message_type



# Step 3: Get Random Knowledge Articles (for query generation)    
@tool
def GetRandomKnowledgeArticles(existing_queries_made: list[str], random_query: str) -> list[str]:
    """New message that needs to test the chatbot are required to be relevant. 
    Use this function to retrieve random knowledge artcile from chatbots knowledgebase. You can further call `NewMessageGeneratorUsingKnowledgeArticles` to create message or query using these retrieved random knowledge.
    
    DO NOT RUN this for message_types such as 'A random irrelevant OOD question', 'personal info question', or 'social message like greetings'.
    
    Params:
    - existing_queries_made (list[str]): Keep track of all the previous queries to get Knowledge Articles that you have made. The new query should be unique.
    - random_query (str): The new query that would try to fetch some document from knowledge base.
    """
    
    return [i.page_content for i in get_relevant_article_snippets(random_query, 1)]


# Step 4: Generate New Message using Knowledge Articles
@tool
def NewMessageGeneratorUsingKnowledgeArticles(message_type_to_generate_new_message: str, new_message: str, conversation_id: str) -> str:
    """Using the user_profile, new_message_type, chat history, and random documents from knowledgebase, Generate a new message that would be sent to the Chatbot for requesting a response.
        The `message_type_to_generate_new_message` is what we have already generated in the previous step.
        The goal of the new message is to stress test the Customer's Chatbot system.
        Therefore, try to generate the new_message based on the provided message_type_to_generate_new_message.
        follow_up messages should always have the conversation_id same as what is returned by the previous call.
        Queries should usually be generated based on random documents from knowledgebase unless `message_type_to_generate_new_message` is testing out-of-domain questions or a social query.
    """
    return new_message


# Step 5: Get ChatBot Response
@tool
def GetChatBotResponse(message: str, conversation_id: str) -> dict[str, str]:
    """This tool is used to interact with the Customer's Chatbot.
       Params:
         - message (str): The user message that the Chatbot needs to provide a response for. This can also be a follow-up question. Follow-up questions need to have the same conversation_id.
         - conversation_id (str): this is an unique identifier. Choose a new conversation_id string to start a new conversation, or use an existing one to continue a previous conversation. Conversations have a maximum number of 10 turns. 
    """
    
    if not conversation_id:
        conversation_id = str(uuid4())
    
    # chatbot_url = "https://container-rag-challenge.ambitiousbeach-b93588ea.eastus.azurecontainerapps.io/chatbot1"

    payload = json.dumps({
      "conversation_id": conversation_id,
      "message": message
    })
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {chatbot_api_key}'
    }

    response = requests.request("POST", chatbot_url, headers=headers, data=payload)
    if "response" not in json.loads(response.text):
        print(json.loads(response.text))
        print(payload)
    
    while "response" not in json.loads(response.text) or json.loads(response.text)["response"] == "Maximum number of conversation turns reached":
        conversation_id = str(uuid4())
        payload = json.dumps({
          "conversation_id": conversation_id,
          "message": message
        })
        response = requests.request("POST", chatbot_url, headers=headers, data=payload)
        if "response" not in json.loads(response.text):
            print(json.loads(response.text))
            print(payload)
        
    return {"response": json.loads(response.text)["response"], "conversation_id": str(uuid4()) if not conversation_id else conversation_id}

# Step 6: Evaluate ChatBot Response
@tool
def EvaluateChatBotResponse(reformulated_message: str, response: str, message_type: str) -> dict[str, Any]:
    """Evaluate the Response provided by the Customer's Chatbot.
     Params:
          - reformulated_message (str): If the message provided to Chatbot is not a follow-up message, `reformulated_message` is same as `message`.
                                            If it is a follow-up message, rephrase the message such that, it can be understood standalone without the previous message.
          - response (str): The response of the Customer's Chatbot the User message.
          - message_type (str): The current message_type that we stress-testing for.
    """
    resp_eval_1 = intermediate_llm.invoke([
        (
            "system",
            """You are given a User Message and a System Response. Evaluate the System Response based on the following factors and return a dictionary of bool flags for the specified metrics.
                Metrics to evaluate:
                - `is_no_answer`: Whether the provided System Response couldn't find an answer to the User Message.
                - `is_helpful_answer`: Whether the provided System Response constructively helps the User.
                - `irrelevant_answer`: Whether the provided System Response is not addressing the context of User Message.
                - `non_actionable_user_like_response`: Whether the System Response is trying to achieve what is not possible for a Chatbot. eg: trying to send an email.
                
                Make sure the response you return can be loaded with json.loads() and is a dictionary.
                DO NOT WRAP THEM in ```json``` or any other code block.
            """
        ),
        (
            "user",
            f"""User Message: {reformulated_message}
            System Response: {response}"""
        )
    ])
    
    relevant_snippets_for_response = get_relevant_article_snippets(response)
    relevant_snippets_for_message = get_relevant_article_snippets(reformulated_message)
    
    formatted_results = "Relevant Knowledgebase Articles:\n"
    for idx, chunk in enumerate(relevant_snippets_for_response+relevant_snippets_for_message):
        formatted_results += f"<id: {id}>\n\n Snippet: {chunk.page_content}\n\n<id: {id}>"
        formatted_results += "------------------"
    
    resp_eval_2 = intermediate_llm.invoke([
        (
            "system",
            """You are given a User Message, System Response, and Knowledgebase Articles. 
                The System Response that was provided by an another system is what you should be evaluating.
                The System Response was provided to the User Message using the Knowledgebase Articles.
                Identify the claims that are made in the System Response but is missing in the knowledgebase Article.
                
                Respond with a dictionary of following values:
                - all_claims_made_in_system_response (str): all the significant claims in the System Response.
                - missing_claims_in_knowledgebase (str): all the claims that the System Response contains but not present in knowledgebase.
                - missing_claims_in_knowledgebase_rate (float): A non-factual rate from 0 to 1 based on `missing_claims_in_knowledgebase`. 1.0 signifies all of the claims are missing from knowledgebase. 0.0 singifies all of the claims are present in knowledbase.
                
                Make sure the response you return can be loaded with json.loads() and is a dictionary.
                DO NOT WRAP THEM in ```json``` or any other code block.
            """
        ),
        (
            "user",
            f"""User Message: {reformulated_message}
            System Response: {response}
            {formatted_results}"""
        )
    ])
    
    resp_eval_3 = intermediate_llm.invoke([
        (
            "system",
            """You are given a User Message, System Response, and Knowledgebase Articles. 
                The System Response that was provided by an another system is what you should be evaluating.
                The System Response was provided to the User Message using the Knowledgebase Articles.
                Identify claims that are relevant to the User Message available in the Knowledgebase Article but is not present in System Response.

                Respond with a dictionary of following values:
                - all_claims_made_in_system_response (str): all the significant claims in the System Response.
                - missing_relevant_claims_in_system_response (str): all the claims that the knowledgebase contains which are relevant to the User Message but are not present in System Response. Capture only the significant ones.
                - missing_relevant_claims_in_system_response_rate (float): A non-faithful rate from 0 to 1 based on `missing_relevant_claims_in_system_response`. 1.0 signifies all of the claims in knowledgebase are missing from System Response. 0.0 singifies all of the claims from knowledgebase are present in System Response.
            
                Make sure the response you return can be loaded with json.loads() and is a dictionary.
                DO NOT WRAP THEM in ```json``` or any other code block.
            """
        ),
        (
            "user",
            f"""User Message: {reformulated_message}
            System Response: {response}
            {formatted_results}"""
        )
    ])
    
    def parse_json(response):
        try:
            return json.loads(response)
        except Exception as e:
            return literal_eval(response)
        except Exception as e:
            return {}
    
    resp_eval_4 = get_response_personality(response)
    resp_eval_1 = parse_json(resp_eval_1.content)
    resp_eval_2 = parse_json(resp_eval_2.content)
    resp_eval_3 = parse_json(resp_eval_3.content)
    resp_eval_2.pop("all_claims_made_in_system_response")
    resp_eval_2.pop("missing_claims_in_knowledgebase")
    resp_eval_3.pop("all_claims_made_in_system_response")
    resp_eval_3.pop("missing_relevant_claims_in_system_response")
    
    personality_score = resp_eval_4["Agreeableness"] * 0.4 + resp_eval_4["Agreeableness"] * 0.05 + resp_eval_4["Extroversion"] * 0.4 + resp_eval_4["Neuroticism"] * 0.075 + resp_eval_4["Openness"] * 0.075
    
    eval_df.loc[eval_df.shape[0]] = [
        reformulated_message,
        response,
        message_type,
        resp_eval_1["is_no_answer"],
        resp_eval_1["is_helpful_answer"],
        resp_eval_1["irrelevant_answer"],
        resp_eval_1["non_actionable_user_like_response"],
        resp_eval_2["missing_claims_in_knowledgebase_rate"],
        resp_eval_3["missing_relevant_claims_in_system_response_rate"],
        personality_score,
        len(response.split(" "))
    ]
    
    return {**resp_eval_1, **resp_eval_2, **resp_eval_3, **resp_eval_4}

# Step 2-6: Repeat the above steps for each test case
# ----------------------------------------------------
# Step 7: End and Create New Topic
@tool
def EndAndCreateNewTopic() -> str:
    """End the current topic and start a new topic with the Chatbot. The function returns a new conversation_id that should be used for new topic messages.
    """
    return str(uuid4())

# Step 8: Print Evaluation Results
# @tool
# def PrintEvaluationResults() -> pd.DataFrame:
#     """Print the Evaluation Results for the current session. Called after all the tests are completed.
#     """
#     # print(st.dataframe(eval_df))
#     return st.dataframe(eval_df)


agent_llm = ChatOpenAI(
    model=agent_model, #"gpt-3.5-turbo-0125",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming=True,
    api_key=openai_api_key
)

tools = [GetUserProfile, TestCaseGenerator, GetRandomKnowledgeArticles, NewMessageGeneratorUsingKnowledgeArticles, GetChatBotResponse, EvaluateChatBotResponse]


MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Chatbot evaluator system that simulates specific user interactions on the Chatbot to stress test the system.
                Your goal is to continously interact with the Chatbot system by generating new messages and switching topics.
                Try Evaluating the System response after even QA turn. Use the evaluation response to improve the next set of messages.
                You can call `TestCaseGenerator` to decide what type of test to run next.
                
                Follow the below sequence of operation:
                1. Run `GetUserProfile` to get user profile. This would give you more information about the user that we are simulating.
                2. Run `TestCaseGenerator` and figure out what test case to evaluate the system.
                3. Run `GetRandomKnowledgeArticles` to retrieve some knowledge articles that chatbot has access to.
                3. Run `NewMessageGeneratorUsingKnowledgeArticles` to generate the actual message that can be passed to the Chatbot System for response.
                4. Run `GetChatBotResponse` to fetch the chatbot response.
                5. Run `EvaluateChatBotResponse` to fetch the quality of system generated response
                6. Run `TestCaseGenerator` to check if we need to perform another test. If it returns empty, quit.
                7. Run `EndAndCreateTopic` if you need to create another topic that is not a follow-up.
                8. Repeat from Step 2 until `TestCaseGenerator` returns empty. 
                
                You are free to create new topic and generate new questions. 
                You do not need to trigger `GetUserProfile` more than once.
                Always remember the test cases generated, even if you create new topic.
                
                You should NOT STOP untill `TestCaseGenerator` returns empty.
                Run `GetUserProfile` the first time to know more about user input and for every new topic or conversation, run `TestCaseGenerator`.
            """,
        ),
        # MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# Run `PrintEvaluationResults` to print the evaluation results once all the conversation and tests are complete

llm_with_tools = agent_llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=50, return_intermediate_steps=True)

with st.form(key="form"):
    if not enable_custom:
        "Run Eval with one of the prefilled options, or enter your API Key in the sidebar to run your own eval."

    prefilled_options = []
    for full_name, name in SAVED_SESSIONS.items():
        prefilled_options.append(name + f"({full_name.split('_url_')[1].replace('.pickle', '')})")
    prefilled = st.selectbox("Preliminary User group profile", sorted(prefilled_options)) or ""
    user_input = ""

    if enable_custom:
        user_input = st.text_input("Or, ask with your own profile")
        enabled_storing = st.checkbox("Save this for later playback")
    if not user_input:
        user_input = prefilled
        user_input = user_input.split(f"({chatbot_selection})")[0]
    submit_clicked = st.form_submit_button("Submit Eval Run")

output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🥷")
    st_callback = StreamlitCallbackHandler(answer_container)
    cfg = RunnableConfig()
    cfg["callbacks"] = [st_callback]
    if user_input + "_url_" + chatbot_selection + ".pickle" not in SAVED_SESSIONS:
        st_callback_store = CapturingCallbackHandler()
        cfg["callbacks"].append(st_callback_store)

    # If we've saved this question, play it back instead of actually running LangChain
    # (so that we don't exhaust our API calls unnecessarily)
    if user_input + "_url_" + chatbot_selection + ".pickle" in SAVED_SESSIONS:
        # session_name = SAVED_SESSIONS[user_input]
        session_name = user_input + "_url_" + chatbot_selection + ".pickle"
        session_path = Path(__file__).parent / "runs" / session_name
        print(f"Playing saved session: {session_path}")
        answer = playback_callbacks([st_callback], str(session_path), max_pause_time=2)
        st.dataframe(pd.read_pickle(str(session_path).replace(".pickle", "_eval_df.pickle")))
    else:
        answer = agent_executor.invoke({"input": f"Initial User profile: {user_input}"}, cfg)
        if enabled_storing:
            location = Path(__file__).parent / "runs" / f"{user_input}_url_{chatbot_selection}.pickle"
            st_callback_store.dump_records_to_file(str(location))
            
            eval_df.to_pickle(str(location).replace(".pickle", "_eval_df.pickle"))
        st.dataframe(eval_df)
            

    answer_container.write(answer["output"])