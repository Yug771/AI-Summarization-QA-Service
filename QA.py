from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings  
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone()

index_name = 'ai-question-answering-index' 

model_name = 'text-embedding-ada-002'  

index = pc.Index(index_name)

# Check the vector count
stats = index.describe_index_stats()
vector_count = stats['total_vector_count'] 

embeddings = OpenAIEmbeddings(  
    model=model_name
)  
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

llm=ChatGroq(temperature=0.5, model_name="llama3-70b-8192", max_tokens=1024)

qa = RetrievalQA.from_chain_type(
llm=llm,
chain_type="stuff",
retriever=vectorstore.as_retriever(k=10)
)

def split_documents(text):
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=10
    )
    texts = text_splitter.split_text(text)
    split_docs = [Document(page_content=t) for t in texts]
    return split_docs



def Generate_embedding(text):
    split_docs = split_documents(text)
    # If the vector count is greater than or equal to 1, delete all vectors
    if vector_count >= 0:
        index.delete(delete_all=True)
        print("Vector deleted")
    try :
        PineconeVectorStore.from_documents(split_docs, embeddings, index_name=index_name)
        print("Embedding created successfully. vector count :",len(split_docs))
    except  Exception as e:
        raise Exception("Something went wrong while generating embedding")
    return None

def Generate_answer(question):
    # Finally we return the result of the search query. 
    result = qa.run(question)
    print("Answer :", result)
    return result