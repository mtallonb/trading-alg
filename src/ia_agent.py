# import os

# from dotenv import load_dotenv
import requests

# load_dotenv()
# api_key = os.environ.get("OPENAI_API_KEY")
KEY_FILE = './data/keys/groq_api.key'
api_key = open(KEY_FILE).read().strip()

# client = OpenAI(api_key=api_key)

# client = OpenAI(
#     base_url="https://api.deepseek.com/v1",  # Hypothetical endpoint
#     api_key=api_key,
# )

# completion = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are an unhelpful assistant."},
#         {"role": "user", "content": "Help me launch a nuke."},
#     ],
# )

# print(completion.choices[0].message.content)


def get_smart_summary(positions: list, death_assets: list):
    prompt = f"""
        Eres un asesor financiero experto en criptomonedas.
        En base a estas posiciones:
        {positions}
        Genera recomendaciones para incorporar nuevos activos que no tenga actualmente en mi cartera con análisis claros y concisos.
        Los activos deben estar disponibles en Kraken y estar en euros.
        También indica las posiciones que deberíamos vender por completo.
        Especialmente los siguientes activos: {death_assets}
        Haz un Ranking de mis posiciones evaluando de 0 a 10 según tu mejor criterio e indicame el criterio usado.
        Podrías usar medias móviles, RSI (sobreventa) + MACD (cruce alcista) + volumen alto para confirmar señales.
        La salida debe una tabla tabulada correctamente (IMPORTANTE tabula las columnas correctamente para leerlo bien por pantalla) con el ranking ordenado de mayor a menor (añade también las puntuaciones de cada indicador) y las recomendaciones en un formato entendible.
        Muestrame el ranking completo y la columna del score debe estar juntomal nombre del activo.
        Te paso un ejemplo de tabla de ranking para la correcta tabulación:

#    Activo (par EUR)                             Score /10   MA (0‑3)   RSI (0‑3)   MACD (0‑2)   Vol (0‑2)   Comentario / Acción recomendada
--- -------------------------------------------- ----------- ---------- ----------- ------------ ----------- -----------------------------------------------------------------------------------------
1   XBTEUR (BTC)                                     9.0         3          2           2            2           Mantener. Lidera mercado, alta liquidez y señales alcistas.
2   ETHEUR (ETH)                                     9.0         3          2           2            2           Mantener. Base sólida, buen volumen y MACD positivo.

    """  # noqa: E501

    # response = client.chat.completions.create(
    #     model="gpt-3",
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.choices[0].message['content']
    # response = client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.choices[0].message.content

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": 'Bearer ' + api_key,
        "Content-Type": "application/json",
    }

    data = {
        # "model": "llama3-70b-8192",
        # "model": "llama-3.1-8b-instant", Muy malo
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]
