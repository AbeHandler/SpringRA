from requests.exceptions import HTTPError
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from urllib.parse import quote
from tqdm import tqdm
import time
import requests
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
S2_KEY = os.getenv("S2_KEY")

@dataclass
class SearchResult:
    year: int
    title: str
    text: str
    id_: str
    query: str = None
    citationCount: int = None
    fieldsOfStudy: str = None
    venue: str = None
    authors: str = None

    def to_json(self) -> dict:
        """Serializes the object to JSON"""
        return self.__dict__

    def to_jsonl(self) -> str:
        """Serializes the object to str"""
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(json_data: str):
        """Deserializes JSON to a SearchResult object"""
        data = json.loads(json_data)
        return SearchResult(**data)


class SemanticScholarAPI(object):
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    # &minCitationCount=200

    def __init__(self, sleepseconds=1):
        self.sleepseconds = sleepseconds

    def bulk_search(self, venue):
        assert len(venue) > 2
        venue_encoded = quote(venue)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search/bulk?venue={venue_encoded}"
        headers = {"x-api-key": S2_KEY}
        response = requests.get(
                url,
                headers=headers,
                timeout=30,
            )
        data = response.json()
        total = data["total"]
        out = []
        for _ in data["data"]:
            out.append(_)

        token = data["token"]

        iters = int(total/1000)

        # 9/30/25 totak seems unreliable so I will look 10x which can get 10000 papers

        for i in tqdm(range(10)):
            time.sleep(self.sleepseconds)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search/bulk?venue={venue_encoded}&token={token}"
            headers = {"x-api-key": S2_KEY}
            response = requests.get(
                    url,
                    headers=headers,
                    timeout=30,
                )
            data = response.json()
            if "data" in data:
                for _ in data["data"]:
                    out.append(_)
            if "token" in data:
                if data["token"] is None:
                    return out
        return out

    def batch_get_paper_details(self, paper_ids, fields='embedding.specter_v2,references,abstract'):
        """
        Fetch paper details in batch using the Semantic Scholar batch API.

        Parameters:
        - paper_ids (list): List of paper IDs to fetch
        - fields (str): Comma-separated fields to retrieve

        Returns:
        - list: List of paper detail dictionaries
        """
        time.sleep(1)
        headers = {"x-api-key": S2_KEY}
        assert type(fields) == str
        response = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': fields},
            json={"ids": paper_ids},
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    def get_papers_citations(self, paper_id):

        def process_data(data, query_id):
            data = data["data"]
            tmp = []
            for _ in data:
                _ = _["citingPaper"]
                _["citesId"] = query_id
                tmp.append(_)
            return tmp

        out = []
        for j in range(500):
            offset = j * 100
            url = f'''https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=venue&offset={offset}'''
            headers = {"x-api-key": S2_KEY}
            response = requests.get(
                    url,
                    headers=headers,
                    timeout=30,
                )
            data = response.json()
            print(response)
            if "data" in data:
                out = out + process_data(data, paper_id)
                if len(data["data"]) < 100:
                    return out
                else:
                    time.sleep(1)
            else:
                pass
        return out

    def get_authors_papers(self, authorid):
        # get up to 500 papers from an author
        out = []
        for offset in [0, 100, 200, 300, 400]:
            url = f"https://api.semanticscholar.org/graph/v1/author/{authorid}/papers?offset={offset}"
            headers = {"x-api-key": S2_KEY}
            response = requests.get(
                    url,
                    headers=headers,
                    timeout=30,
                )
            data = response.json()
            if "data" in data:
                out = out + data["data"]
            else:
                print(data)
            time.sleep(2)
        return out

        # https://api.semanticscholar.org/graph/v1/author/1741101/papers

    def get_author_id(self, author):
        # e.g. authorId
        from urllib.parse import quote
        author_encoded = quote(author)
        url = f"https://api.semanticscholar.org/graph/v1/author/search?query={author_encoded}"
        headers = {"x-api-key": S2_KEY}
        response = requests.get(
                url,
                headers=headers,
                timeout=30,
            )

        data = response.json()["data"]

        # e.g. [{'authorId': '2263872612', 'name': 'Jeffrey K Wooldridge " K'} ... ]
        ids = [o["authorId"] for o in data] 
        response = response.json()["data"]
        r = requests.post(
            'https://api.semanticscholar.org/graph/v1/author/batch',
            params={'fields': 'name,hIndex,citationCount'},
            json={"ids":ids}
        )
        data = r.json()

        if len(data) == 0:
            return []

        data.sort(key=lambda x: int(x["citationCount"]), reverse=True)

        return data[0]


    @classmethod
    def get_headers(cls):

        headers = {"x-api-key": S2_KEY}

        return headers

    @classmethod
    def build_result_object(self, json_object, query):
        s2id = json_object["paperId"]
        title = json_object["title"]
        abstract = json_object["abstract"]
        year = json_object["year"]
        citationCount = json_object['citationCount']
        fieldsOfStudy = json_object['fieldsOfStudy']
        authors = ", ".join([o["name"] for o in json_object['authors']])
        venue = json_object["venue"]
        return SearchResult(
            year=year,
            title=title,
            text=abstract, 
            venue=venue, 
            authors=authors,
            id_=s2id, fieldsOfStudy=fieldsOfStudy,
            query=query, citationCount=citationCount
        )

    @classmethod
    def multiple_searches(cls, query, start_year=1980, end_year=2023, iterations=10, batch_size=100):
        out = []
        for i in range(iterations):
            rs = cls.search(query, start_year, end_year, offset=i * batch_size)  # Call the search method with the calculated offset
            rs = [o for o in rs if isinstance(o, dict)]
            out += rs  # Accumulate the results
        return out

    @classmethod
    def search(cls, query, start_year=1980, end_year=2023, offset=0, minCitationCount=10):
        headers = SemanticScholarAPI.get_headers()
        try:
            response = requests.get(
                cls.BASE_URL,
                params={
                    "query": query,
                    "year": f"{start_year}-{end_year}",
                    "limit": 100,
                    "offset": offset,
                    "minCitationCount": minCitationCount,
                    "fieldsOfStudy": ["Business", "Economics"],
                    "fields": "title,year,abstract,citationCount,fieldsOfStudy,authors,venue",
                },
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()

            try:
                json_data = response.json()
            except json.JSONDecodeError:
                print(f"JSON decoding failed for query '{query}'")
                return []

            data = json_data.get("data", [])
            if not data:
                print(f"No data returned for query '{query}'")
                return []

            out = [SemanticScholarAPI.build_result_object(obj, query=query).to_json() for obj in data]
            return out

        except requests.exceptions.HTTPError as e:
            return f"HTTPError for query '{query}': {e}"
        except requests.exceptions.RequestException as e:
            return f"RequestException for query '{query}': {e}"
        return []

    @classmethod
    def search_for_id(cls, query):
        headers = SemanticScholarAPI.get_headers()
        response = requests.get(cls.BASE_URL, params={"query": query}, headers=headers, timeout=30)

        # Handle potential errors
        response.raise_for_status()

        dt = response.json()["data"]
        if len(dt) > 0:
            return dt[0]["paperId"]
        else:
            return None

    @classmethod
    def jaccard_similarity(cls, str1, str2):
        """ """
        # Tokenize the strings by splitting them on whitespace.
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())

        # Calculate the intersection and union of the token sets.
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        # Avoid division by zero if both strings are empty.
        if not union:
            return 1.0 if not intersection else 0.0

        # Calculate Jaccard Similarity.
        return len(intersection) / len(union)

    @classmethod
    def get_id_from_title(cls, title):
        search_url = f'{cls.BASE_URL}?query="{title}"'
        headers = cls.get_headers()

        try:
            response = requests.get(search_url, headers=headers, timeout=30)
            response.raise_for_status()  # Check for HTTP errors

            if "data" not in response.json():
                return "UNK"
            data = response.json()["data"]
            if len(data) > 0:
                id_ = data[0].get("paperId", None)
                title_for_id = cls.get_title_by_id(id_)
                if cls.jaccard_similarity(title.lower(), title_for_id.lower()) > 0.85:
                    return id_
                else:
                    return f"Could not find ID for title {title}"

        except requests.RequestException as e:
            print(f"Request error: {e}")
        except requests.HTTPError as e:
            print(f"HTTP error: {e}")
        except Exception as e:
            print(cls, title)
            print(f"An unexpected error occurred: {e}")

        return None

    @classmethod
    def paper_title_search(cls, title):
        # https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_title_search
        # Behaves similarly to /paper/search, but is intended for retrieval of a single paper based on closest title match to given query.
        url = f'''https://api.semanticscholar.org/graph/v1/paper/search/match?query={quote(title)}'''
        headers = {"x-api-key": S2_KEY}
        response = requests.get(
                url,
                headers=headers,
                timeout=30,
            )

        class APIError(Exception):
            """General exception for API call failures"""
            pass

        if response.status_code == 429:
            raise APIError("Rate limit exceeded (HTTP 429).")

        response.raise_for_status()
        data = response.json()

        if "data" not in data:
            raise APIError(f"Could not find {title}")

        return data["data"][0]["paperId"]


    @classmethod
    def get_abstract_by_id(cls, paper_id):
        try:
            headers = SemanticScholarAPI.get_headers()
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=abstract"
            paper_url = url
            response = requests.get(paper_url, params={}, headers=headers, timeout=30)

            # Handle potential errors
            response.raise_for_status()

            return response.json()["abstract"]
        except HTTPError:
            return "UNK"

    @classmethod
    def get_title_by_id(cls, paper_id):
        try:
            headers = SemanticScholarAPI.get_headers()
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=title"
            paper_url = url
            response = requests.get(paper_url, params={}, headers=headers, timeout=30)

            # Handle potential errors
            response.raise_for_status()

            return response.json()["title"]
        except HTTPError:
            return "UNK"

    @classmethod
    def get_year_by_id(cls, paper_id):

        headers = SemanticScholarAPI.get_headers()
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=year"
        paper_url = url
        response = requests.get(paper_url, params={}, headers=headers, timeout=30)

        # Handle potential errors
        response.raise_for_status()

        return response.json()["year"]


    @classmethod
    def get_authors_by_id(cls, paper_id):
        """
        Retrieves the list of authors for a given paper ID from the Semantic Scholar API.

        Parameters:
        - paper_id (str): The Semantic Scholar ID of the paper for which to retrieve authors.

        Returns:
        - list[dict]: A list of dictionaries containing author details (e.g., name, authorId).
        """
        headers = SemanticScholarAPI.get_headers()
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=authors"
        
        response = requests.get(url, headers=headers, timeout=30)

        # Handle potential errors
        response.raise_for_status()

        # Extract authors from the response JSON
        authors = response.json().get("authors", [])
        
        # Return author details (name and authorId)
        return ",".join([author["name"] for author in authors])

    @classmethod
    def get_citation_count_by_id(cls, paper_id):

        headers = SemanticScholarAPI.get_headers()
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=citationCount"
        paper_url = url
        response = requests.get(paper_url, params={}, headers=headers, timeout=30)

        # Handle potential errors
        response.raise_for_status()

        return response.json()["citationCount"]

    @classmethod
    def get_s2id_from_corpus_id(cls, corpusID):
        url = f'https://api.semanticscholar.org/graph/v1/paper/CorpusID:{corpusID}'
        response = cls.make_get_request(url)
        return response.json()["paperId"]

    @classmethod
    def get_references(cls, paper_id):
        """
        Retrieves the references for a given paper ID from the Semantic Scholar API.

        Parameters:
        - paper_id (str): The Semantic Scholar ID of the paper for which to retrieve references.

        Returns:
        - list[dict]: A list of dictionaries containing reference details (e.g., title, paperId, year).
        """
        try:
            headers = cls.get_headers()
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=references"
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Extract references from the response JSON
            references = response.json().get("references", [])
            if not references:
                print(f"No references found for paper ID {paper_id}")
                return []

            # Build a list of dictionaries for each reference
            out = []
            for ref in references:
                ref_paper_id = ref.get("paperId")
                ref_title = ref.get("title", "No title")
                ref_year = ref.get("year", "No year")
                ref_citation_count = ref.get("citationCount", 0)
                ref_fields_of_study = ref.get("fieldsOfStudy", [])
                
                out.append({
                    "paperId": ref_paper_id,
                    "title": ref_title,
                    "year": ref_year,
                    "citationCount": ref_citation_count,
                    "fieldsOfStudy": ref_fields_of_study,
                })

            return out

        except requests.exceptions.HTTPError as e:
            print(f"HTTPError for paper ID '{paper_id}': {e}")
        except requests.exceptions.RequestException as e:
            print(f"RequestException for paper ID '{paper_id}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return []

    @classmethod
    def make_get_request(cls, url, params=None, headers=None, timeout=30):
        """
        Makes a GET request to the specified URL with optional parameters, headers, and timeout.

        Parameters:
        - url (str): The URL to which the GET request is sent.
        - params (dict, optional): The parameters to send in the query string. Defaults to {}.
        - headers (dict, optional): The headers for the request. If None, uses default headers from get_headers().
        - timeout (int, optional): The timeout for the request in seconds. Defaults to 30.

        Returns:
        - response (requests.Response): The response object from the requests library.
        """
        if headers is None:
            headers = cls.get_headers()

        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad responses

        return response

def get_cites_for_berente():
    api = SemanticScholarAPI()

    # Research Commentary - Data-Driven Computationally Intensive Theory Development
    cites = api.get_papers_citations("ef1318a7008570ab5f9cfbfe14db624d0db8d5d4")
    IDS = []
    for cite in cites:
        IDS.append(cite['paperId'].strip('"'))

    time.sleep(1)
    details = api.batch_get_paper_details(paper_ids = IDS, fields="title,abstract,year,citations,references")
    for ino, i in enumerate(details):
        details[ino]["Ncitations"] = len(details[ino]["citations"])

    df = pd.DataFrame(details)
    df = df.sort_values("Ncitations")
    with open("data/berente_2019.jsonl", "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    get_cites_for_berente()
    