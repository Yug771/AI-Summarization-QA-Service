# summarization.py

from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
#-------------------------------------------------Extractive-Import------------------------------------#
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from nltk.tokenize import sent_tokenize
import torch
from summarizer import Summarizer


tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")

model = LongT5ForConditionalGeneration.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps", return_dict_in_generate=True).to("cpu")


from dotenv import load_dotenv
load_dotenv()

class Summarization:
    def __init__(self, model):
        self.model = model

    def summarize(self, text):
        raise NotImplementedError("Summarize method not implemented!")

    def get_chunk_size(self):
        raise NotImplementedError("Get chunk size method not implemented!")

    def split_documents(self, text):
        chunk_size = self.get_chunk_size()
        text_splitter = TokenTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=10
        )
        texts = text_splitter.split_text(text)
        split_docs = [Document(page_content=t) for t in texts]
        return split_docs

    def needs_splitting(self):
        raise NotImplementedError("Needs splitting method not implemented!")


class AbstractiveSummarization(Summarization):
    def summarize(self, text):
        if self.needs_splitting():
            split_docs = self.split_documents(text)
            summaries = self.model_summarize(split_docs)
            return summaries
        else:
            return self.model_summarize(text)

    def get_chunk_size(self):
        chunk_sizes = {
            "OpenAI": 128000,
            "Gemma": 1000,
            "Gemini": 100000,
            "Llama": 5500,
            "Mixtral": 4500
        }
        return chunk_sizes.get(self.model, 1000)  # Default chunk size

    def needs_splitting(self):
        # Models that do not need splitting
        no_split_models = ['Pegasus']
        return self.model not in no_split_models

    def model_summarize(self, text):
        # Implement model-specific summarization logic here
        if self.model == "OpenAI":
            return self.openai_summarize(text)
        elif self.model == "Gemma":
            return self.gemma_summarize(text)
        elif self.model == "Gemini":
            return self.gimini_summarize(text)
        elif self.model == "Llama":
            return self.llama_summarize(text)
        elif self.model == "Mixtral":
            return self.mixtral_summarize(text)
        elif self.model == "Pegasus":
            return self.pegasus_summarize(text)
        else:
            raise ValueError("Unknown model!")

    def openai_summarize(self, split_docs):
        # Define prompt
        prompt_template = """Write a concise summary in 90 words of the following:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        print(f"\n\n Number of split_docs for summary:{len(split_docs)}\n\n")
        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        summary = stuff_chain.invoke(split_docs)["output_text"]
        return summary

    def gemma_summarize(self, split_docs):
        llm = ChatGroq(model_name='gemma2-9b-it')

        # Create a prompt template for individual chunk summarization
        individual_prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text:\n\n{text}\n\nSummary:"
        )

        # Create a prompt template for the final summary
        final_prompt = PromptTemplate(
            input_variables=["text"],
            template="Combine the following summaries into a coherent overall summary:\n\n{text}\n\n directly give the summary without any extra prefix token"
        )

        # Create the summarization chain for individual chunks
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=individual_prompt,
            combine_prompt=final_prompt,
            verbose=True
        )

        # Generate the final summary
        final_summary = chain.invoke(split_docs)

        return final_summary['output_text']

    def gimini_summarize(self, text):
        llm = ChatGoogleGenerativeAI(model="gemini-pro",
                 temperature=0.7, top_p=0.85)

        # To query Gemini
        llm_prompt_template = """Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:"""
        llm_prompt = PromptTemplate.from_template(llm_prompt_template)
        doc_prompt = PromptTemplate.from_template("{page_content}")
        stuff_chain = (
            # Extract data from the documents and add to the key `text`.
            {
                "text": lambda docs: "\n\n".join(
                    format_document(doc, doc_prompt) for doc in docs
                )
            }
            | llm_prompt         # Prompt for Gemini
            | llm                # Gemini function
            | StrOutputParser()  # output parser
        )
        return stuff_chain.invoke(text)

    def llama_summarize(self, split_docs):
        llm = ChatGroq(model_name='llama3-8b-8192')

        # Create a prompt template for individual chunk summarization
        individual_prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text:\n\n{text}\n\nSummary:"
        )

        # Create a prompt template for the final summary
        final_prompt = PromptTemplate(
            input_variables=["text"],
            template="Combine the following summaries into a coherent overall summary and directly give the point wise summary in response without any prefix token:\n\n{text}\n\n summary:"
        )

        # Create the summarization chain for individual chunks
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=individual_prompt,
            combine_prompt=final_prompt,
            verbose=True
        )
        # Generate the final summary
        final_summary = chain.invoke(split_docs)

        return final_summary['output_text']
        
    def mixtral_summarize(self, split_docs):
        llm = ChatGroq(model_name='mixtral-8x7b-32768')

        # Create a prompt template for individual chunk summarization
        individual_prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text:\n\n{text}\n\nSummary:"
        )

        # Create a prompt template for the final summary
        final_prompt = PromptTemplate(
            input_variables=["text"],
            template="Combine the following summaries into a coherent overall summary:\n\n{text}\n\n directly give the summary without any extra prefix token"
        )

        # Create the summarization chain for individual chunks
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=individual_prompt,
            combine_prompt=final_prompt,
            verbose=True
        )

        # Generate the final summary
        final_summary = chain.invoke(split_docs)

        return final_summary['output_text']

    def pegasus_summarize(self, text):
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cpu")
        sequences = model.generate(input_ids,max_new_tokens=256).sequences

        summary = tokenizer.batch_decode(sequences)
        summary = summary[0][6:-4]

        #----Handling sentence cut-of----#
        sentences = summary.split('.')
    
        # Remove any empty sentences created by split
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Check if the last sentence ends with a period
        if not summary.strip().endswith('.'):
            # If it doesn't end with a period, remove the last sentence
            sentences.pop()
        
        # Join the remaining sentences back into a single string
        cleaned_summary = '. '.join(sentences) + '.'
        return cleaned_summary


class ExtractiveSummarization(Summarization):
    def summarize(self, text):
        if self.needs_splitting():
            split_docs = self.split_documents(text)
            summaries = [self.model_summarize(doc) for doc in split_docs]
            return " ".join(summaries)
        else:
            return self.model_summarize(text)

    def get_chunk_size(self):
        chunk_sizes = {
            "LuhnSumy": 2000,
            "Pegasus": 3000
        }
        return chunk_sizes.get(self.model, 1000)  # Default chunk size

    def needs_splitting(self):

        return False

    def model_summarize(self, text):
        # Implement model-specific summarization logic here
        if self.model == "LuhnSumy":
            return self.luhn_summarize(text)
        elif self.model == "BERT":
            return self.BERT_summarize(text)
        else:
            raise ValueError("Unknown model!")

    def luhn_summarize(self, text):
        # Complete quote sentences
        number_of_sentences=3
        sentences = sent_tokenize(text)
        # For Strings
        parser = PlaintextParser.from_string(text,Tokenizer("english"))
        # Using KL
        summarizer = LuhnSummarizer()
        #Summarize the document with 4 sentences
        summary = summarizer(parser.document,number_of_sentences)

        summary_text=[]
        for sentence in summary:
            summary_text.append(str(sentence))
        return summary_text[0]    

    def BERT_summarize(self, text):
        model = Summarizer()
        result = model(text, min_length=60)
        summary = ''.join(result)
        sentences = summary.split('.')
    
        # Remove any empty sentences created by split
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Check if the last sentence ends with a period
        if not summary.strip().endswith('.'):
            # If it doesn't end with a period, remove the last sentence
            sentences.pop()
        
        # Join the remaining sentences back into a single string
        cleaned_summary = '. '.join(sentences) + '.'
        return cleaned_summary
    