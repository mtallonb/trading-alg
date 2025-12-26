# import os

# from dotenv import load_dotenv

from textwrap import dedent

import requests

from google import genai

VALIDS_AGENTS = ['groq', 'gemini', 'openai']

# load_dotenv()


def get_smart_summary(positions: list, death_assets: list, ia_agent: str = 'groq') -> str | None:
    """
    Generates a financial summary using the specified AI model.

    Args:
        positions: A list of current portfolio positions.
        death_assets: A list of assets to consider selling.
        ia_agent: The AI model to use ('groq', 'gemini', or 'openai').

    Returns:
        The response from the AI model as a string, or an error message.
    """
    prompt = _generate_prompt(positions, death_assets)

    api_calls = {
        'groq': _call_groq_api,
        'gemini': _call_gemini_api,
        'openai': _call_openai_api,
    }

    api_function = api_calls.get(ia_agent)

    if api_function:
        return api_function(prompt=prompt)
    else:
        return f"Error: IA agent '{ia_agent}' is not supported. Available agents are: {list(api_calls.keys())}"


# ----------------- Internal functions --------------------------------


def _generate_prompt(positions: list, death_assets: list) -> str:
    """Generates the prompt for the financial advisor AI."""

    """
    Meter todos los trades e indicarle cuando salir de los muertos. Voy a abrir una posición de compra o venta 
    de acuerdo a la volatilidad en tal activo te paso mi ranking. Critica mi ranking.
    """

    prompt = f"""
        You are an expert financial advisor specializing in cryptocurrencies.
        Based on these current positions:
        {positions}

        1.  **Generate Recommendations**: Recommend new assets to add to my portfolio. These assets must not be in my current positions. Provide clear and concise analysis for each recommendation. They must be available on Kraken and denominated in EUR.
        2.  **Identify Sells**: Indicate which positions should be sold completely, paying special attention to the following underperforming assets: {death_assets}.
        3.  **Rank My Portfolio**: Create a ranking of my current positions, evaluating them from 0 to 10 based on your expert judgment. Explain the criteria used for the score. You should use technical indicators like Moving Averages, RSI (oversold), MACD (bullish crossover), and high volume to confirm signals.
        4.  **Format the Output**: The output must be a correctly tabulated table. It is VERY IMPORTANT to tabulate the columns correctly for proper screen display. The ranking should be sorted from highest to lowest score. The score column must be next to the asset name.

        Here is an example of the required ranking table format:

        #    Asset (EUR pair)                             Score /10   MA (0-3)   RSI (0-3)   MACD (0-2)   Vol (0-2)   Comment / Recommended Action
        --- -------------------------------------------- ----------- ---------- ----------- ------------ ----------- -----------------------------------------------------------------------------------------
        1   XBTEUR (BTC)                                     9.0         3          2           2            2           Hold. Market leader, high liquidity, and bullish signals.
        2   ETHEUR (ETH)                                     9.0         3          2           2            2           Hold. Solid foundation, good volume, and positive MACD.

        IMPORTANT: Remember to tabulate the output correctly.
    """  # noqa: E501
    return dedent(prompt)


def _read_api_key(key_file: str) -> str | None:
    """Reads an API key from a file."""

    try:
        with open(key_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file not found at {key_file}")
        return None


def _call_openai_api(prompt: str) -> str | None:
    """Calls the OpenAI API and returns the response content."""

    api_key = _read_api_key('./data/keys/openai_api.key')
    if not api_key:
        return "OpenAI API key not found."

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error calling OpenAI API: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing OpenAI API response: {e}"


def _call_gemini_api(prompt: str) -> str | None:
    """Calls the Google Gemini API and returns the response content."""

    api_key = _read_api_key('./data/keys/gemini_api.key')
    if not api_key:
        return "Gemini API key not found."

    try:
        client = genai.Client(api_key=api_key)
        model = 'gemini-2.5-flash'
        print("Sending prompt to Gemini...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"


def _call_groq_api(prompt: str) -> str | None:
    """Calls the Groq API and returns the response content."""

    api_key = _read_api_key('./data/keys/groq_api.key')
    if not api_key:
        return "Groq API key not found."

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f'Bearer {api_key}'},
            json={
                # "model": "llama3-70b-8192",  # Using a recommended model
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error calling Groq API: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing Groq API response: {e}"
