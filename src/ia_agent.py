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
        Eres un asesor financiero experto en cripto y acciones.
        En base a estas posiciones:
        {positions}
        Genera recomendaciones para incorporar nuevos activos que no tenga actualmente en mi cartera con análisis claros y concisos.
        Los activos deben estar disponibles en Kraken y estar en euros.
        También indica las posiciones que deberíamos vender por completo.
        Especialmente los siguientes activos: {death_assets}
        Haz un Ranking de mis posiciones evaluando de 0 a 10 según tu mejor criterio e indicame el criterio usado.
        Podrías usar medias móviles, RSI (sobreventa) + MACD (cruce alcista) + volumen alto para confirmar señales.
        La salida debe ser un json con el ranking (añade también las puntuaciones de cada indicador) y las recomendaciones en un formato entendible.
        Muestrame el ranking completo.
        Smart summary:
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
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]
