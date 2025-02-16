from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings

# Set DeepSeek API Key

base_url = "https://api.aimlapi.com/v1"
api_key="0f33d8a8b7714460bc4b8335b66d217a"
# Initialize OpenAI client
api = OpenAI(api_key=api_key, base_url=base_url)

# Generate 300 Dummy Messages with Severity Levels
severities = ["High", "Medium", "Low"]
messages = [
    {"message": f"Disaster Alert {i}", "severity": random.choice(severities)}
    for i in range(300)
]
df = pd.DataFrame(messages)


# Twitter API Credentials (Replace with your actual credentials)
TWITTER_API_KEY = "your-api-key"
TWITTER_API_SECRET = "your-api-secret"
TWITTER_ACCESS_TOKEN = "your-access-token"
TWITTER_ACCESS_SECRET = "your-access-secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

def fetch_tweets(keyword, count=10):
    """Fetch real-time disaster-related tweets"""
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode="extended").items(count)
    return [{"User": tweet.user.screen_name, "Tweet": tweet.full_text} for tweet in tweets]

# Streamlit UI
st.title("🚨 Real-Time Disaster Analysis with Twitter API")

keyword = st.text_input("Enter disaster keyword (e.g., Earthquake, Flood, Hurricane):", "Earthquake")
if st.button("Fetch Tweets"):
    with st.spinner("Fetching live data..."):
        tweets = fetch_tweets(keyword)
        df = pd.DataFrame(tweets)
        st.dataframe(df)






# PDF Processing for Chatbot Context (Pre-Provided PDF)
pdf_path = "Natural Disaster Safety Manual.pdf"  # Ensure this file is in the same directory as app.py

pdf_reader = PdfReader(pdf_path)
raw_text = ""
for page in pdf_reader.pages:
    raw_text += page.extract_text() + "\n"

# Convert to Embeddings for Retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(raw_text)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(texts, embeddings)
retriever = vector_db.as_retriever()

# LangGraph-Powered Q&A System
chat_model = ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url=base_url)
qa = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever)

# Chatbot UI
st.subheader("🤖 AI-Powered Disaster Chatbot")
user_query = st.text_input("Ask the chatbot:")

if user_query:
    response = qa.run(user_query)
    st.write("**Chatbot Response:**", response)


# Voice Recognition for Non-English Users
st.subheader("🎙️ Voice Recognition (Speech-to-Text)")

if st.button("Start Recording"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    try:
        recognized_text = recognizer.recognize_google(audio)
        st.write("**Recognized Text:**", recognized_text)
    except sr.UnknownValueError:
        st.write("Sorry, could not understand.")
    except sr.RequestError:
        st.write("Could not request results. Check your internet connection.")

# Disaster Guide Dropdown
st.subheader("🌪️ Disaster Preparedness Guide")

disaster_options = {
    "Wildfire": {
        "steps": [
            "Evacuate if ordered.",
            "Keep emergency supplies ready.",
            "Close all doors and windows."
        ],
        "video": "https://www.youtube.com/watch?v=OCjl6tp8dnw"
    },
    "Earthquake": {
        "steps": [
            "Drop, Cover, and Hold On.",
            "Stay indoors until shaking stops.",
            "Move away from windows."
        ],
        "video": "https://www.youtube.com/watch?v=BLEPakj1YTY"
    },
    "Flood": {
        "steps": [
            "Move to higher ground.",
            "Avoid walking or driving through floodwaters.",
            "Stay tuned to emergency alerts."
        ],
        "video": "https://www.youtube.com/watch?v=43M5mZuzHF8"
    }
}

selected_disaster = st.selectbox("Select a disaster type:", list(disaster_options.keys()))

if selected_disaster:
    st.write("### 🛠 Steps to Follow:")
    for step in disaster_options[selected_disaster]["steps"]:
        st.write(f"- {step}")

    st.write("📺 [Watch Video Guide]({})".format(disaster_options[selected_disaster]["video"]))

st.write("🚀 Stay prepared and stay safe!")
