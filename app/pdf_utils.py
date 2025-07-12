import re

import pypdf


def extract_text_from_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_food_plan(text):
    # Dividir o texto em seções de refeições usando palavras-chave comuns
    meal_sections = re.split(
        r"\n(CAFÉ DA MANHÃ|ALMOÇO|LANCHE I|JANTAR|PRE TREINO|TREINO):?",
        text,
        flags=re.IGNORECASE,
    )
    meal_dict = {}
    # Iterar sobre as seções encontradas
    for i in range(1, len(meal_sections), 2):
        meal_name = meal_sections[i].strip().upper()
        meal_content = meal_sections[i + 1].strip()
        # Separar linhas para facilitar a extração
        lines = re.split(r"\n|(?<=\d)(?=[A-Z])", meal_content)
        items = []
        for line in lines:
            # Tentar extrair alimento, medida e quantidade com regex mais flexível
            match = re.match(
                r"([\w\s\(\)\/\-\+\.,]+?)\s*(\d+\s*(?:unidade|porção|copo|xícara|colher|concha|fatia|gr|ml|kg|g|mg|l|unidades|porções|prato|fios|sopa|gramas|ml))",
                line,
                re.IGNORECASE,
            )
            if match:
                food = match.group(1).strip()
                measure = match.group(2).strip()
                # Quantidade pode estar junto da medida, separar se necessário
                quantity = measure
                items.append({"food": food, "measure": measure, "quantity": quantity})
        meal_dict[meal_name] = items
    return meal_dict


def extract_physical_measures(text):
    # Extrai as datas das avaliações
    dates = re.findall(r"\d{2}/\d{2}/\d{4}", text)

    # Define as medidas a serem extraídas (ordem conforme aparecem no texto)
    measure_names = [
        "Circunferência cintura",
        "Circunferência quadril",
        "Circunferência braço",
        "Circunferência coxa",
        "Circunferência pant.",
        "Circunferência peito",
        "Tríceps",
        "Peito",
        "Subaxilar",
        "Subescapular",
        "Abdominal",
        "Supra ilíaca",
        "Coxa",
        "Massa magra",
        "Massa gorda",
        "% G",
        "Peso",
        "Massa muscular",
        "Qualidade muscular",
        "% gordura",
        "Classificação física",
        "Gordura visceral",
        "Idade metabólica",
        "Taxa metabólica basal",
        "% água",
        "IMC",
    ]

    # Regex para capturar os valores após cada medida
    pattern = re.compile(
        r"(Circunferência cintura|Circunferência quadril|Circunferência braço|Circunferência coxa|Circunferência pant\.|Circunferência peito|Tríceps|Peito|Subaxilar|Subescapular|Abdominal|Supra ilíaca|Coxa|Massa magra|Massa gorda|% G|Peso|Massa muscular|Qualidade muscular|% gordura|Classificação física|Gordura visceral|Idade metabólica|Taxa metabólica basal|% água|IMC)\s+([^\n]+)"
    )

    measures = {}
    for match in pattern.finditer(text):
        name = match.group(1)
        values = match.group(2).strip().split()
        measures[name] = values

    return {"dates": dates, "measures": measures}


if __name__ == "__main__":
    # Exemplo de uso:
    evaluation_text = extract_text_from_pdf(
        "pdfs/3ª AVALIAÇÃO NUTRICIONAL - Arthur Bernardo Assumpção Pinto  11-04-2025.pdf"
    )
    plan_text = extract_text_from_pdf(
        "pdfs/3ª PLANO ALIMENTAR - Arthur Bernardo Assumpção Pinto  11-04-2025.pdf"
    )
    medidas = extract_physical_measures(evaluation_text)
    plano = extract_food_plan(plan_text)
    breakpoint()

    print("Medidas Físicas:", medidas)
    print("Plano Alimentar:", plano)
