### LLMs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

geminiflash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
)
# geminipro = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0.1,
# )
gpt4o = ChatOpenAI(model="gpt-4o", temperature=0.1)
claude37 = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0.1)
claude37thinking = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    max_tokens=18000,
    temperature=1,
    thinking={"type": "enabled", "budget_tokens": 16000},
)
