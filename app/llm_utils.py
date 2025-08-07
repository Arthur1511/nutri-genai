import os
import json
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Carrega as variáveis de ambiente do arquivo .env (ex: GOOGLE_API_KEY)
load_dotenv()


def get_vector_store(text: str):
    """
    Cria um banco de dados vetorial (vector store) a partir de um texto.

    Este processo envolve três etapas principais:
    1. Dividir o texto de entrada em pedaços (chunks) menores.
    2. Gerar embeddings (vetores numéricos) para cada chunk usando um modelo do Google.
    3. Indexar esses chunks e seus embeddings em um banco de dados FAISS para
       busca rápida de similaridade.

    Args:
        text: O texto completo a ser indexado.

    Returns:
        Um objeto FAISS que funciona como um banco de dados vetorial.

    Raises:
        ValueError: Se a chave de API do Google não for encontrada nas variáveis de ambiente.
    """
    # 1. Dividir o texto em chunks para processamento pelo modelo de embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # 2. Configurar o modelo de embeddings do Google
    # Certifique-se de que a variável de ambiente GOOGLE_API_KEY está definida.
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY não encontrada. Por favor, configure no arquivo .env."
        )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 3. Criar o Vector Store com FAISS, que indexa os chunks e seus embeddings
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_store


def get_conversational_chain():
    """
    Cria e retorna uma 'chain' de conversação para responder a perguntas.

    A chain é construída com um modelo de linguagem da Google (Gemini) e um
    template de prompt customizado que instrui o modelo a agir como um
    assistente de nutrição, respondendo com base no contexto fornecido.

    Returns:
        Uma 'chain' do LangChain pronta para ser executada.
    """
    # TODO: Mover este template para um arquivo de configuração ou para o `prompts.py`
    # para melhor organização e reutilização.
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

    # Configura o modelo de linguagem (LLM) que será usado para gerar as respostas
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Cria o template de prompt a partir da string
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Carrega a 'chain' de QA (Question Answering) do tipo "stuff",
    # que "enfia" (stuffs) todos os documentos de contexto no prompt.
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def extract_structured_data(text: str) -> dict:
    """
    Usa um LLM para extrair dados estruturados (JSON) de um texto.

    Esta função envia o texto bruto de documentos nutricionais para um LLM
    configurado para retornar uma resposta no formato JSON. O prompt guia o
    modelo para extrair informações específicas como avaliações físicas e
    planos alimentares.

    Args:
        text: O texto bruto extraído dos PDFs.

    Returns:
        Um dicionário Python com os dados estruturados. Retorna um dicionário
        vazio se a extração ou o parsing do JSON falhar.
    """
    
    # Configura um modelo LLM potente (Gemini 1.5 Flash) e o instrui a
    # retornar a resposta diretamente no formato JSON.
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,  # Temperatura 0 para respostas mais determinísticas
        generation_config={
            "response_mime_type": "application/json",
        }
    )

    # TODO: Mover este template para um arquivo de configuração ou para o `prompts.py`.
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

    # Cria a 'chain' de extração usando a sintaxe de pipe (LCEL)
    chain = prompt | llm
    
    # Invoca a 'chain' com o texto de entrada
    response = chain.invoke({"input_text": text})
    
    # O LLM retorna um objeto de mensagem, cujo conteúdo é uma string JSON.
    # É necessário fazer o parsing dessa string para um dicionário Python.
    try:
        # A resposta do LLM pode vir com marcações de código (```json ... ```),
        # então limpamos a string antes de fazer o parsing.
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, AttributeError):
        # Em caso de erro no parsing ou se a resposta não tiver o atributo 'content',
        # retornamos um dicionário vazio para evitar que a aplicação quebre.
        return {}
