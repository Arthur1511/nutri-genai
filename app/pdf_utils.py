import io
from typing import List
import fitz  # PyMuPDF

def get_pdf_text(pdf_files: List[io.BytesIO]) -> str:
    """
    Extrai e concatena o texto de uma lista de arquivos PDF.

    A função itera sobre uma lista de arquivos PDF em memória (geralmente
    provenientes de um uploader de arquivos do Streamlit), abre cada um
    com a biblioteca PyMuPDF (fitz), extrai o texto de todas as páginas e
    retorna uma única string com todo o conteúdo. Um separador é adicionado
    ao final de cada página para manter alguma estrutura.

    Args:
        pdf_files: Uma lista de objetos de arquivo em memória (bytes), como os
                   retornados por `st.file_uploader(accept_multiple_files=True)`.

    Returns:
        Uma string única contendo o texto concatenado de todos os PDFs.
    """
    full_text = ""
    for pdf_file in pdf_files:
        # Garante que a leitura do arquivo comece do início.
        # Isso é importante pois o objeto pode ter sido lido antes.
        pdf_file.seek(0)

        # Abre o PDF a partir do stream de bytes em memória
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            # Itera sobre cada página do documento
            for page in doc:
                # Extrai o texto da página e o adiciona à string completa
                full_text += page.get_text()
                # Adiciona um separador para indicar o fim de uma página
                full_text += "\n--- FIM DA PÁGINA ---\n"

    return full_text