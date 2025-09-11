from serpapi import GoogleSearch
from serpapi import SerpApiClient
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os


load_dotenv()  # picks up OPENAI_* variables from .env if present

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("serpapi")

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def google_search(query: str) -> list[dict]:
    """
    Run a Google search via SerpAPI and return the organic results.
    
    Parameters
    ----------
    query : str
        The search query string (e.g., "Coffee")
    
    Returns
    -------
    list[dict]
        The list of organic search results from Google.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY") # please set your api key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")
