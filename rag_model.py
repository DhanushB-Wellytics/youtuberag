import os

os.environ["OPENAI_API_KEY"] = 
os.environ["PINECONE_API_KEY"] = 
os.environ["PINECONE_API_ENV"] = 


from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)


from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()


from langchain.prompts import ChatPromptTemplate

template = """
You are an AI assistant, trained to provide understandable and accurate information about pharmacogenomics and drugs.
You will base your responses on the context and information provided. Output both your answer and a score of how confident you are,
 and also cite the references. Also provide the source of the chunks of the documents used for response.
If the information related to the question is not in the context and or in the information provided in the prompt, 
you will say 'I don't know'.
You are not a healthcare provider and you will not provide medical care or make assumptions about treatment.


Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


from langchain_openai.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()


from langchain_pinecone import PineconeVectorStore

pinecone = PineconeVectorStore(embedding=embeddings, index_name="pdfs-rag")

retriever = pinecone.as_retriever()


from langchain_core.runnables import RunnableParallel, RunnablePassthrough

setup = RunnableParallel(context=retriever, question=RunnablePassthrough())


import numpy as np

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_file_path = os.path.join(script_dir, 'Questions.csv')

Questions = pd.read_csv(csv_file_path)

questions = Questions["Questions"].to_list()


cont = []
for question in questions:
    cont.append([docs.page_content for docs in retriever.invoke(question)])
Questions["Context"] = cont

chain = setup | prompt | model | parser

resp = []

n = len(questions)
i = 1

for question in questions:
    resp.append(chain.invoke(question))
    Questions["Response"] = resp + [np.nan] * (n - i)
    Questions.to_csv(csv_file_path)
    i += 1
