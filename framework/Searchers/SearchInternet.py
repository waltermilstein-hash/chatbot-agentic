"""
Internet search client that generates grounded answers.
"""
import asyncio

import openai

from config.Constants import GPT_5_1, KEY_OPEN_AI


class WebSearchClient:
    """
    Class to search for information on the internet and generate answers.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the search client with the OpenAI API key.

        Args:
            api_key: OpenAI API key (optional if present in environment variables)
        """
        if api_key:
            openai.api_key = api_key

    async def search_and_answer_web(
            self,
            question: str,
            context_message: str = "No relevant information was found in our local knowledge base"
    ) -> str:
        """
        Search the internet and answer the question.
        v2.0: Migrated to GPT-5 + web_search_preview tool (replaces gpt-5.6-search-preview).

        Args:
            pregunta: The user's question (kept name for cross-module callers)
            mensaje_contexto: Explanatory message about why the internet is being searched

        Returns:
            str: Answer based on an internet search with full disclaimers
        """

        prompt_internet = f"""
    {context_message} Searching for up-to-date information on the internet to answer:

    Question: "{question}"

    Formatting instructions:
    - Provide a useful, clear and accurate answer based on current information
    - If the answer contains comparable data, organize it as a table
    - Use ONLY basic HTML without CSS, styles or classes
    - For tables use only: <table>, <tr>, <th>, <td>
    - For lists use: <ul>, <ol>, <li>
    - For paragraphs use: <p>
    - For titles use: <h3>, <h4>
    - Do NOT include web links - instead mention names of stores, sites or platforms
    - Keep the HTML clean and semantic

    Content instructions:
    - Search for and use up-to-date, relevant information
    - If it is an opinion question, offer balanced perspectives
    - Keep a friendly and professional tone
    - IMPORTANT: At the start of the answer include: <p><small>🌐 <strong>Information extracted from the internet:</strong></small></p>
    - At the end, BEFORE the sources table, always include these disclaimers:
    <p><small>⚠️ <strong>Important warnings:</strong></small></p>
    <ul><small>
    <li>The data is extracted from the internet and, for greater accuracy, it is recommended that the user review it</li>
    <li>Published information may not be current and is subject to constant change</li>
    <li>It is recommended to verify directly with official sources before making decisions</li>
    </small></ul>

    Instructions for sources and places:
    - MANDATORY: At the end, ALWAYS indicate the specific sources in a table with the columns: Information source, Where to find it
    - In "Information source": mention specific sources where the information was extracted from (e.g.: "Official websites", "News portals", "Specialized platforms")
    - In "Where to find it": mention specific names of sites or platforms WITHOUT links (e.g.: "Amazon, MercadoLibre, Falabella", "CNN, BBC, Reuters", "Official brand sites")
    - If you cannot identify specific sources, indicate "Multiple internet sources" and "Verify on official sites for the topic queried"
    - For deals/discounts add: "It is recommended to verify current availability and prices"
    - Do NOT make up links - only mention names of well-known places to search

    Answer in clean HTML:
    """

        try:
            client = openai.AsyncOpenAI(api_key=KEY_OPEN_AI)

            # GPT-5 + Responses API with web_search (renamed from web_search_preview)
            # Ref: https://platform.openai.com/docs/guides/tools-web-search
            response = await asyncio.wait_for(
                client.responses.create(
                    model=GPT_5_1,
                    tools=[{"type": "web_search"}],
                    reasoning={"effort": "low"},  # web search does not need deep reasoning
                    input=prompt_internet,
                    max_output_tokens=5000,
                ),
                timeout=160.0
            )

            # The Responses API returns output_text directly
            content = response.output_text or ""

            return content.strip() if content else "<p>No answer was obtained from the search.</p>"

        except asyncio.TimeoutError:
            return "<p>Sorry, the internet search took too long. Please try again.</p>"
        except Exception as e:
            return (
                f"<p>Sorry, I could not look up up-to-date information at this time.</p>"
                f"<p><small>Error: {str(e)}</small></p>"
            )

# Usage example:
if __name__ == "__main__":
    # Create the search client instance
    client = WebSearchClient()

    # Run a search
    question = "What are the best-selling mobile phones in 2024?"
    answer = client.search_and_answer_web(question)
    print(answer)
