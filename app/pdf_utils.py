import re
import fitz  # PyMuPDF
from typing import List
import io

def get_pdf_text(pdf_files: List[io.BytesIO]) -> str:
    """
    Extrai texto de uma lista de arquivos PDF enviados via Streamlit.

    Args:
        pdf_files: Uma lista de objetos de arquivo em memória (do st.file_uploader).

    Returns:
        Uma string única contendo o texto concatenado de todos os PDFs.
    """
    full_text = ""
    for pdf_file in pdf_files:
        pdf_file.seek(0)  # Garante que o ponteiro do arquivo esteja no início
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text()
                full_text += "\n--- FIM DA PÁGINA ---\n"
    return full_text

def extract_food_plan(text):
    # (Sua função original com regex)
    meal_sections = re.split(
        r"\n(CAFÉ DA MANHÃ|ALMOÇO|LANCHE I|JANTAR|PRE TREINO|TREINO):?",
        text,
        flags=re.IGNORECASE,
    )
    meal_dict = {}
    for i in range(1, len(meal_sections), 2):
        meal_name = meal_sections[i].strip().upper()
        meal_content = meal_sections[i + 1].strip()
        lines = re.split(r"\n|(?<=\d)(?=[A-Z])", meal_content)
        items = []
        for line in lines:
            match = re.match(
                r"([\w\s\(\)\/\-\+\.,]+?)\s*(\d+\s*(?:unidade|porção|copo|xícara|colher|concha|fatia|gr|ml|kg|g|mg|l|unidades|porções|prato|fios|sopa|gramas|ml))",
                line,
                re.IGNORECASE,
            )
            if match:
                food = match.group(1).strip()
                measure = match.group(2).strip()
                quantity = measure
                items.append({"food": food, "measure": measure, "quantity": quantity})
        meal_dict[meal_name] = items
    return meal_dict

def extract_physical_measures(text):
    # (Sua função original com regex)
    dates = re.findall(r"\d{2}/\d{2}/\d{4}", text)
    measure_names = [
        "Circunferência cintura", "Circunferência quadril", "Circunferência braço",
        "Circunferência coxa", "Circunferência pant.", "Circunferência peito",
        "Tríceps", "Peito", "Subaxilar", "Subescapular", "Abdominal",
        "Supra ilíaca", "Coxa", "Massa magra", "Massa gorda", "% G", "Peso",
        "Massa muscular", "Qualidade muscular", "% gordura", "Classificação física",
        "Gordura visceral", "Idade metabólica", "Taxa metabólica basal", "% água", "IMC",
    ]
    pattern = re.compile(
        r"(Circunferência cintura|Circunferência quadril|Circunferência braço|Circunferência coxa|Circunferência pant\.|Circunferência peito|Tríceps|Peito|Subaxilar|Subescapular|Abdominal|Supra ilíaca|Coxa|Massa magra|Massa gorda|% G|Peso|Massa muscular|Qualidade muscular|% gordura|Classificação física|Gordura visceral|Idade metabólica|Taxa metabólica basal|% água|IMC)\s+([^\n]+)"
    )
    measures = {}
    for match in pattern.finditer(text):
        name = match.group(1)
        values = match.group(2).strip().split()
        measures[name] = values
    return {"dates": dates, "measures": measures}