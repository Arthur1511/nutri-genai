import streamlit as st
from pdf_utils import get_pdf_text
from llm_utils import get_vector_store, get_conversational_chain, extract_structured_data
from dashboard_utils import create_evolution_charts

def main():
    st.set_page_config(layout="wide", page_title="NutriGenAI", page_icon="🤖")

    # --- INICIALIZAÇÃO DO SESSION STATE ---
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "structured_data" not in st.session_state:
        st.session_state.structured_data = None

    st.title("🤖 NutriGenAI: Análise Inteligente de Planos Alimentares")

    # --- SIDEBAR PARA UPLOAD E PROCESSAMENTO ---
    with st.sidebar:
        st.header("Upload de Arquivos")
        uploaded_files = st.file_uploader(
            "Faça o upload dos seus arquivos PDF (avaliação física, plano alimentar):",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Processar Arquivos"):
                with st.spinner("Extraindo texto dos PDFs..."):
                    raw_text = get_pdf_text(uploaded_files)
                
                with st.spinner("Analisando documentos com IA para extrair dados..."):
                    st.session_state.structured_data = extract_structured_data(raw_text)

                with st.spinner("Criando banco de dados vetorial para o chat..."):
                    st.session_state.vector_store = get_vector_store(raw_text)
                
                st.session_state.messages = []
                st.success("Arquivos processados com sucesso!")
                st.rerun()

    # --- DASHBOARD DE ANÁLISE ---
    st.header("Dashboard de Análise")
    if st.session_state.structured_data:
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Faça uma pergunta sobre seus documentos..."):
        if not st.session_state.vector_store:
            st.warning("Por favor, processe os arquivos primeiro.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                docs = st.session_state.vector_store.similarity_search(prompt)
                chain = get_conversational_chain()
                response = chain(
                    {"input_documents": docs, "question": prompt},
                    return_only_outputs=True
                )
                response_text = response["output_text"]
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()