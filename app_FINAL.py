import streamlit as st
import pandas as pd 
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
import os
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"   
)
llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(
    page_title="InsightIQ",
    page_icon="📊",
    layout="wide"
)
with st.sidebar:
    st.title("📊 InsightIQ")
    st.markdown("---")

    st.header(" Dashboard")
    st.success("✅ CSV Upload")
    st.subheader("📄 PDF Analysis")

    pdf_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
    )
    if pdf_file is not None:
        pdf_bytes = pdf_file.getvalue() 


        with open("temp.pdf", "wb") as f:
         f.write(pdf_bytes)

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        st.success(
         f"✅ PDF loaded successfully! Pages: {len(documents)}"
        )
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

        docs = text_splitter.split_documents(documents)

        st.success(f"✅ PDF split into {len(docs)} chunks")
        embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
        docs,
        embeddings
        )

        st.success("✅ PDF embeddings and vector database created!")
        st.subheader("📚 Ask Questions from PDF")

        pdf_question = st.text_input(
        "Ask a question about your PDF",
        key="pdf_question"
         )

        if st.button("Ask PDF"):
         try:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

            relevant_docs = retriever.invoke(pdf_question)

            context = "\n\n".join(
                doc.page_content
                for doc in relevant_docs
            )

            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        You are a helpful document analyst.

                        Answer the question only using the
                        provided PDF context.

                        PDF Context:
                        {context}

                        If the answer is not present in the
                        context, say that the information is
                        not available in the PDF.
                        """
                    },
                    {
                        "role": "user",
                        "content": pdf_question
                    }
                ]
            )

            st.subheader("🤖 Answer")

            st.write(
                response.choices[0].message.content
            )

         except Exception as e:
            st.error(f"Error: {e}")
        

    

    
    st.info("📊 Data Summary")

    st.info("📈 Charts")

    st.info("🤖 AI Insights (Coming Soon)")

st.title("📊 InsightIQ")
st.subheader("AI Business Intelligence Copilot")

st.caption("Upload a CSV file to explore, visualize, and analyze your data with AI.")

uploaded_file = st.file_uploader(
    "📂 Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    df = pd.read_csv(uploaded_file)

    conn = sqlite3.connect("sales.db")

    db = SQLDatabase.from_uri("sqlite:///sales.db")

    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    agent = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        handle_parsing_errors=True
    )

    df.to_sql(
        "sales_data",
        conn,
        index=False,
        if_exists="replace"
    )
    st.subheader("📋 Dataset Preview")

    st.dataframe(df)
    st.subheader("📊 Dataset Summary")

    st.write("Rows :", df.shape[0])

    st.write("Columns :", df.shape[1])

    st.write("Missing Values :", df.isnull().sum().sum())
    st.header("📈 Visualizations")

chart = st.selectbox(
    "Select a Chart",
    ["Bar Chart", "Pie Chart", "Histogram"]
)
x_axis = st.selectbox(
    "Select X-Axis",
    df.columns
)
numeric_columns = df.select_dtypes(include="number").columns

y_axis = st.selectbox(
    "Select Y-Axis",
    numeric_columns
)

if chart == "Bar Chart":

    st.subheader("📊 Sales by Product")

    fig = px.bar(
    df,
    x=x_axis,
    y=y_axis,
    title=f"{y_axis} by {x_axis}"
)

    st.plotly_chart(fig)
elif chart == "Pie Chart":

    st.subheader("🥧 Sales Distribution by Product")

    pie_fig = px.pie(
        df,
        names=x_axis,
        values=y_axis,
        title="Sales Distribution"
    )

    st.plotly_chart(pie_fig)
elif chart == "Histogram":

    st.subheader("📈 Sales Distribution")

    hist_fig = px.histogram(
        df,
        x=y_axis,
        nbins=10
    )

    st.plotly_chart(hist_fig)
st.header("📋 Data Explorer")

tab1, tab2, tab3, tab4 = st.tabs([
    "Columns",
    "Data Types",
    "Missing Values",
    "Summary"
])

with tab1:
    st.write(list(df.columns))

with tab2:

    data_types = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(data_types)

with tab3:
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    st.dataframe(missing_values)

with tab4:
    st.dataframe(df.describe())
    st.header("🔍 Filter Data")

product_filter = st.selectbox(
    "Select Product",
    ["All"] + list(df["Product"].unique())
)

if product_filter != "All":
    filtered_df = df[df["Product"] == product_filter]
else:
    filtered_df = df

st.dataframe(filtered_df)
st.header("🗄️ SQL Query")

query = st.text_area(
    "Enter SQL Query",
    "SELECT * FROM sales_data"
)

if st.button("Run Query"):

    try:
        result = pd.read_sql_query(
            query,
            conn
        )

        st.dataframe(result)

    except Exception as e:

        st.error(e)

st.header("🤖 AI Copilot")

user_question = st.text_input(
    "Ask a question about your dataset"
)


if st.button("Ask AI"):

    try:

        dataset_info = df.head().to_string()

        response = client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[

                {
                    "role": "system",
                    "content": f"""
                    You are a data analyst.

                    Dataset preview:

                    {dataset_info}

                    Answer questions only using this dataset.
                    """
                },

                {
                    "role": "user",
                    "content": user_question
                }

            ]
        )

        st.write(response.choices[0].message.content)

    except Exception as e:

        st.error(f"Error: {e}")
st.header("🗄️ Automated SQL Generator")

nl_query = st.text_input(
    "Ask your data question in plain English",
    key="sql_question"
)

if st.button("Generate SQL"):

    try:

        schema = ", ".join(df.columns)

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are an expert SQL analyst.

                    Convert the user's question into a valid SQLite query.

                    Database table: sales_data

                    Columns: {schema}

                    Return ONLY the SQL query.
                    Do not explain anything.
                    """
                },
                {
                    "role": "user",
                    "content": nl_query
                }
            ]
        )

        generated_sql = response.choices[0].message.content.strip()

        st.subheader("Generated SQL")
        st.code(generated_sql, language="sql")

        result = pd.read_sql_query(
            generated_sql,
            conn
        )

        st.subheader("Query Result")
        st.dataframe(result)

    except Exception as e:
        st.error(f"Error: {e}")        

    
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📦 Total Products", df["Product"].nunique())

with col2:
    st.metric("💰 Total Sales", f"₹ {df['Sales'].sum():,}")

with col3:
    st.metric("📊 Average Sales", f"₹ {df['Sales'].mean():,.0f}")
    