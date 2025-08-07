import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

def create_evolution_charts(data: Dict[str, Any]) -> List[go.Figure]:
    """
    Cria gráficos de linha da evolução temporal das métricas físicas.

    A função recebe um dicionário de dados estruturados, extrai as medições
    (assessments), transforma-os em um DataFrame do Pandas e, em seguida,
    gera um gráfico de linha para cada unidade de medida encontrada (ex: kg, %).

    Args:
        data: O dicionário de dados estruturados extraído pelo LLM.
              Espera-se que contenha a chave 'assessments'.

    Returns:
        Uma lista de figuras do Plotly (go.Figure), onde cada figura
        representa a evolução das métricas para uma unidade de medida específica.
        Retorna uma lista vazia se não for possível gerar os gráficos.
    """
    # Validação inicial dos dados de entrada
    if not data or "assessments" not in data or not data["assessments"]:
        return []

    # 1. Achatando os dados:
    # Transforma a lista aninhada de avaliações em uma lista simples de dicionários,
    # facilitando a conversão para um DataFrame.
    flat_data = []
    for assessment in data["assessments"]:
        date = assessment.get("date")
        if not date:
            continue  # Pula avaliações sem data
        for metric in assessment.get("metrics", []):
            flat_data.append({
                "date": date,
                "name": metric.get("name"),
                "value": metric.get("value"),
                "unit": metric.get("unit", "N/A")  # Define 'N/A' se a unidade não existir
            })

    if not flat_data:
        return []

    # 2. Criando e limpando o DataFrame do Pandas
    df = pd.DataFrame(flat_data)
    # Converte a coluna de data para o formato datetime, tratando possíveis erros
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
    # Converte a coluna de valor para o formato numérico, tratando possíveis erros
    df["value"] = pd.to_numeric(df["value"], errors='coerce')
    # Remove linhas que tenham valores nulos em colunas essenciais
    df.dropna(subset=["date", "value", "name"], inplace=True)
    # Ordena o DataFrame por data para garantir que as linhas do gráfico sejam desenhadas corretamente
    df.sort_values("date", inplace=True)

    # 3. Agrupando por unidade e criando um gráfico para cada
    charts = []
    unique_units = df["unit"].unique()

    for unit in unique_units:
        # Filtra o DataFrame para conter apenas as métricas da unidade atual
        df_unit = df[df["unit"] == unit]
        
        # Cria o gráfico de linha usando Plotly Express
        fig = px.line(
            df_unit,
            x='date',
            y='value',
            color='name',  # Cria uma linha diferente para cada métrica
            title=f'Evolução das Métricas ({unit})',
            markers=True,  # Adiciona marcadores em cada ponto de dados
            labels={'date': 'Data', 'value': f'Valor ({unit})', 'name': 'Métrica'}
        )
        # Centraliza o título e personaliza o layout da legenda
        fig.update_layout(
            title_x=0.5,
            legend_title_text='Métricas'
        )
        charts.append(fig)

    return charts
