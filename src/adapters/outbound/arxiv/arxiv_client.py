"""
ArXiv API HTTP client.

Handles all communication with the ArXiv API.
"""
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from typing import Optional
from src.config import MAX_ARXIV_RESULTS


def arxiv_query(query: str, max_results: int = MAX_ARXIV_RESULTS) -> str:
    """
    Make HTTP request to ArXiv API and return raw XML response.
    
    Args:
        query: ArXiv search query string
        max_results: Maximum number of results to return
        
    Returns:
        Raw XML response as string
    """
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": query,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"ArXiv API request failed: {e}")


def arxiv_response_parser(xml_string: str) -> pd.DataFrame:
    """
    Parse ArXiv XML response into structured DataFrame.
    
    Args:
        xml_string: Raw XML response from ArXiv API
        
    Returns:
        DataFrame with paper metadata
    """
    try:
        root = ET.fromstring(xml_string)
        
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        
        for entry in root.findall('atom:entry', ns):
            # Extract fields
            arxiv_id = entry.findtext('atom:id', '', ns).split('/abs/')[-1] if entry.findtext('atom:id', '', ns) else ''
            title = entry.findtext('atom:title', '', ns)
            summary = entry.findtext('atom:summary', '', ns)
            published = entry.findtext('atom:published', '', ns)
            updated = entry.findtext('atom:updated', '', ns)
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.findtext('atom:name', '', ns)
                if name:
                    authors.append(name)
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', ns):
                term = category.get('term', '')
                if term:
                    categories.append(term)
            
            # Extract primary category (first one)
            primary_category = categories[0] if categories else ''
            
            # Optional fields
            doi = ''
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'doi':
                    doi = link.get('href', '')
                    break
            
            journal_ref = ''
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'journal-ref':
                    journal_ref = link.get('href', '')
                    break
            
            comment = ''
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'comment':
                    comment = link.get('href', '')
                    break
            
            papers.append({
                'arxiv_id': arxiv_id,
                'title': title.replace('\n', ' ') if title else '',
                'summary': summary.replace('\n', ' ') if summary else '',
                'published': published,
                'updated': updated,
                'authors': authors,
                'categories': categories,
                'primary_category': primary_category,
                'doi': doi,
                'journal_ref': journal_ref,
                'comment': comment
            })
        
        return pd.DataFrame(papers)
    
    except ET.ParseError as e:
        raise RuntimeError(f"Failed to parse ArXiv XML response: {e}")


def get_papers_from_query(query: str, max_results: int = MAX_ARXIV_RESULTS) -> pd.DataFrame:
    """
    Retrieve papers from ArXiv using a search query.
    
    Args:
        query: ArXiv search query string
        max_results: Maximum number of papers to retrieve
        
    Returns:
        DataFrame with paper metadata
    """
    try:
        xml_response = arxiv_query(query, max_results)
        papers_df = arxiv_response_parser(xml_response)
        return papers_df
    except Exception as e:
        print(f"Error retrieving papers: {e}")
        return pd.DataFrame()
