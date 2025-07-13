import os

from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()


def get_vector_store(text: str):
    """
    Cria um banco de dados vetorial a partir de um texto.

    Args:
        text: O texto completo para ser indexado.

    Returns:
        Um objeto FAISS contendo os vetores do texto.
    """
    # 1. Dividir o texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # 2. Configurar os embeddings do Google
    # Certifique-se que a GOOGLE_API_KEY está no seu .env
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY não encontrada. Por favor, configure no arquivo .env."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 3. Criar o Vector Store com FAISS
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_store


def get_conversational_chain():
    """
    Cria e retorna uma chain de conversação para responder perguntas.
    """
    prompt_template = """
    Você é um assistente de nutrição. Sua tarefa é responder perguntas sobre os documentos fornecidos.
    Use o contexto dos documentos para fornecer respostas precisas e úteis.
    Se a resposta não estiver no contexto, diga "A informação não está disponível nos documentos".
    
    Contexto:
    {context}
    
    Pergunta:
    {question}
    
    Resposta:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


import json

def extract_structured_data(text: str) -> dict:
    """
    Usa um LLM para extrair dados estruturados (JSON) de um texto de avaliação nutricional.

    Args:
        text: O texto bruto extraído dos PDFs.

    Returns:
        Um dicionário Python com os dados estruturados.
    """
    
    # Modelo configurado para retornar JSON
    # Usamos um modelo mais recente e potente para essa tarefa complexa
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        generation_config={
            "response_mime_type": "application/json",
        }
    )

    prompt = PromptTemplate(
        template="""
        Analise o seguinte texto de uma avaliação nutricional que contém o histórico de medições.
        O texto pode ter várias colunas de datas. Para cada métrica, extraia o valor correspondente a cada data.
        Estruture a saída em um formato JSON que contenha uma lista de avaliações, onde cada item da lista representa uma data de avaliação.

        Texto para análise:
        --- TEXTO ---
        {input_text}
        --- FIM DO TEXTO ---

        Formato JSON de saída esperado:
        {{
            "assessments": [
                {{
                    "date": "DD/MM/AAAA",
                    "metrics": [
                        {{ "name": "Peso", "value": 80.5, "unit": "kg" }},
                        {{ "name": "% Gordura", "value": 22.1, "unit": "%" }},
                        {{ "name": "Massa Muscular", "value": 35.0, "unit": "kg" }}
                    ]
                }},
                {{
                    "date": "DD/MM/AAAA",
                    "metrics": [
                        {{ "name": "Peso", "value": 78.2, "unit": "kg" }},
                        {{ "name": "% Gordura", "value": 20.5, "unit": "%" }},
                        {{ "name": "Massa Muscular", "value": 35.5, "unit": "kg" }}
                    ]
                }}
            ],
            "meal_plan": {{ 
                "last_update_date": "DD/MM/AAAA",
                "meals": [
                    {{ 
                        "name": "Café da Manhã", 
                        "items": [
                            {{ "food": "Ovo", "quantity": 2, "unit": "unidades" }},
                            {{ "food": "Pão Integral", "quantity": 1, "unit": "fatia" }}
                        ]
                    }}
                ]
            }}
        }}

        Se uma informação não for encontrada, omita a chave ou use um valor nulo.
        O plano alimentar geralmente corresponde à avaliação mais recente.
        """,
        input_variables=["input_text"],
    )

    chain = prompt | llm
    
    response = chain.invoke({"input_text": text})
    
    # O LLM retorna o conteúdo como uma string, precisamos converter para um dicionário Python
    try:
        # A resposta pode vir com markdown (```json ... ```), então limpamos isso
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        # Em caso de erro na conversão, retornamos um dicionário vazio
        # ou podemos tentar uma correção com outro prompt (mais avançado)
        return {}
