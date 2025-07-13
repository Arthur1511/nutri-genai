import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

def create_evolution_charts(data: Dict[str, Any]) -> List[go.Figure]:
    """
    Cria gráficos de linha da evolução temporal das métricas físicas, 
    agrupando-as por unidade de medida.

    Args:
        data: O dicionário de dados estruturados contendo a chave 'assessments'.

    Returns:
        Uma lista de figuras do Plotly, uma para cada unidade de medida.
    """
    if not data or "assessments" not in data or not data["assessments"]:
        return []

    # 1. Achatando os dados em uma lista de dicionários planos
    flat_data = []
    for assessment in data["assessments"]:
        date = assessment.get("date")
        if not date:
            continue
        for metric in assessment.get("metrics", []):
            flat_data.append({
                "date": date,
                "name": metric.get("name"),
                "value": metric.get("value"),
                "unit": metric.get("unit", "N/A")
            })

    if not flat_data:
        return []

    # 2. Criando o DataFrame
    df = pd.DataFrame(flat_data)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
    df["value"] = pd.to_numeric(df["value"], errors='coerce')
    df.dropna(subset=["date", "value", "name"], inplace=True)
    df.sort_values("date", inplace=True)

    # 3. Agrupando por unidade e criando gráficos
    charts = []
    unique_units = df["unit"].unique()

    for unit in unique_units:
        df_unit = df[df["unit"] == unit]
        
        fig = px.line(
            df_unit,
            x='date',
            y='value',
            color='name',
            title=f'Evolução das Métricas ({unit})',
            markers=True,
            labels={'date': 'Data', 'value': f'Valor ({unit})', 'name': 'Métrica'}
        )
        fig.update_layout(
            title_x=0.5,
            legend_title_text='Métricas'
        )
        charts.append(fig)

    return charts
