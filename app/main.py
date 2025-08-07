import streamlit as st
from pdf_utils import get_pdf_text
from llm_utils import get_vector_store, get_conversational_chain, extract_structured_data
from dashboard_utils import create_evolution_charts

def main():
    """
    Função principal que executa a aplicação Streamlit NutriGenAI.

    A aplicação permite o upload de planos alimentares e avaliações físicas em PDF,
    extrai dados estruturados usando IA, exibe um dashboard com a evolução das
    métricas e oferece um chat interativo para tirar dúvidas sobre os documentos.
    """
    st.set_page_config(layout="wide", page_title="NutriGenAI", page_icon="🤖")

    # --- INICIALIZAÇÃO DO SESSION STATE ---
    # O st.session_state é usado para manter o estado da aplicação entre os reruns.
    # Inicializamos as chaves necessárias se elas ainda não existirem.
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None  # Armazena o banco de dados vetorial para o chat
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Armazena o histórico de mensagens do chat
    if "structured_data" not in st.session_state:
        st.session_state.structured_data = None  # Armazena os dados extraídos para o dashboard

    st.title("🤖 NutriGenAI: Análise Inteligente de Planos Alimentares")

    # --- SIDEBAR PARA UPLOAD E PROCESSAMENTO ---
    # A sidebar é usada para ações principais, como o upload de arquivos.
    with st.sidebar:
        st.header("Upload de Arquivos")
        uploaded_files = st.file_uploader(
            "Faça o upload dos seus arquivos PDF (avaliação física, plano alimentar):",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Processar Arquivos"):
                # 1. Extrair texto bruto dos PDFs
                with st.spinner("Extraindo texto dos PDFs..."):
                    raw_text = get_pdf_text(uploaded_files)
                
                # 2. Extrair dados estruturados (JSON) do texto usando um LLM
                with st.spinner("Analisando documentos com IA para extrair dados..."):
                    st.session_state.structured_data = extract_structured_data(raw_text)

                # 3. Criar um banco de dados vetorial para o chat de perguntas e respostas
                with st.spinner("Criando banco de dados vetorial para o chat..."):
                    st.session_state.vector_store = get_vector_store(raw_text)
                
                # 4. Limpar histórico de chat anterior e notificar o sucesso
                st.session_state.messages = []
                st.success("Arquivos processados com sucesso!")
                st.rerun() # st.rerun() recarrega a UI para refletir as mudanças de estado

    # --- DASHBOARD DE ANÁLISE ---
    st.header("Dashboard de Análise")
    if st.session_state.structured_data:
        # Gera e exibe os gráficos de evolução se os dados estiverem disponíveis
        charts = create_evolution_charts(st.session_state.structured_data)
        if charts:
            for chart in charts:
                st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("Não foi possível gerar gráficos de evolução com os dados extraídos.")
    else:
        st.warning("Processe seus arquivos para ver o dashboard.")

    # --- SEÇÃO DE CHAT ---
    st.header("Chat Interativo")

    # Exibe o histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura a nova pergunta do usuário
    if prompt := st.chat_input("Faça uma pergunta sobre seus documentos..."):
        # Verifica se os documentos foram processados antes de permitir o chat
        if not st.session_state.vector_store:
            st.warning("Por favor, processe os arquivos primeiro.")
            st.stop()

        # Adiciona a pergunta do usuário ao histórico e à UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processa a pergunta e obtém a resposta do LLM
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                # Busca documentos similares no banco vetorial
                docs = st.session_state.vector_store.similarity_search(prompt)
                # Cria a cadeia de conversação
                chain = get_conversational_chain()
                # Executa a cadeia com os documentos e a pergunta
                response = chain(
                    {"input_documents": docs, "question": prompt},
                    return_only_outputs=True
                )
                response_text = response["output_text"]
                st.markdown(response_text)
        
        # Adiciona a resposta do assistente ao histórico
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    # Ponto de entrada da aplicação
    main()