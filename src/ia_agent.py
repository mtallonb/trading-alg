from textwrap import dedent

import requests

from google import genai
from groq import Groq

VALID_AGENTS = ['groq', 'gemini', 'openai']


def get_smart_summary(positions: list, death_assets: list, ia_agent: str = 'groq') -> str:
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

    if ia_agent not in VALID_AGENTS:
        return f"Error: IA agent '{ia_agent}' is not supported. Available agents are: {VALID_AGENTS}"

    return api_calls[ia_agent](prompt=prompt)


# ----------------- Internal functions --------------------------------


def _generate_prompt(positions: list, death_assets: list) -> str:
    """Generates the prompt for the financial advisor AI."""

    prompt = f"""
    SYSTEM ROLE:
    You are a Senior Crypto Quantitative Analyst and Financial Advisor. Your expertise covers technical analysis, market sentiment, and Kraken exchange listings.

    INPUT DATA:
    - Current Portfolio Positions: {positions}
    - Critical Underperforming Assets (Priority Sells): {death_assets}
    - Currency Denomination: EUR
    - Exchange: Kraken

    TASK INSTRUCTIONS:
    1.  **Portfolio Ranking & Scoring**:
        Evaluate each current position from 0 to 10. The score MUST be the sum of these technical components:
        - Moving Averages (MA) [0-3 pts]: Trend alignment.
        - RSI [0-3 pts]: Relative strength (oversold conditions preferred).
        - MACD [0-2 pts]: Bullish/Bearish crossovers.
        - Volume [0-2 pts]: Liquidity and signal confirmation.
        Explain your specific criteria for the final score.

    2.  **Strategic Recommendations**:
        Suggest 1-3 new assets available on Kraken (EUR pairs) NOT currently in the portfolio. Provide a concise 1-sentence bullish thesis for each based on current market trends.

    3.  **Sell/Exit Strategy**:
        Identify which assets to liquidate. Focus heavily on {death_assets}. Provide a brief reason for each exit.

    4.  **Formatting Requirements**:
        - Use a Markdown code block for the table to ensure fixed-width alignment.
        - Sort the table by Score in descending order.
        - Map names correctly: XBT=Bitcoin, XDG=Dogecoin, POL=Polygon, AEUR=EOS, DOT=Polkadot.

    REQUIRED TABLE FORMAT EXAMPLE:
    ```
    #   Asset (EUR)      Score/10   MA(3)  RSI(3)  MACD(2)  Vol(2)   Action & Expert Comment
    --- ---------------- ---------- ------ ------- -------- -------- ---------------------------------------
    1   XBTEUR (BTC)     9.5        3.0    2.5     2.0      2.0      Strong Hold. Dominance increasing...
    ```

    IMPORTANT: Ensure the table is perfectly aligned. If an asset is in {death_assets}, its score should reflect its underperformance.
    """  # noqa: E501
    return dedent(prompt)


def _read_api_key(key_file: str) -> str:
    """Reads an API key from a file."""

    try:
        with open(key_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _call_openai_api(prompt: str) -> str:
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
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error calling OpenAI API: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing OpenAI API response: {e}"


def _call_gemini_api(prompt: str) -> str:
    """Calls the Google Gemini API using the GenAI SDK."""

    api_key = _read_api_key('./data/keys/gemini_api.key')
    if not api_key:
        return "Gemini API key not found."

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            # model='gemini-2.5-flash',
            model='gemini-2.5-flash-lite',
            contents=prompt,
        )
        return response.text

    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"


def _call_groq_api(prompt: str) -> str:
    """Calls the Groq API and returns the response content."""

    api_key = _read_api_key('./data/keys/groq_api.key')
    if not api_key:
        return "Groq API key not found."

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior investment strategist. "
                    "Provide sharp, data-driven advice on my portfolio",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
            timeout=30,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error calling Groq API: {e!s}"
