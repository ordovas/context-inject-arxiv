"""
REST API routes for paper retrieval.

Simple wrapper around run_arxiv_agent() for HTTP access.
"""
from flask import Blueprint, request, jsonify
from pydantic import ValidationError
import asyncio

from src.application.usecases.arxiv import run_arxiv_agent
from src.config import LLM_MODEL
from src.adapters.inbound.rest_api.models import (
    PaperSearchRequest,
    PaperSearchResponse,
    PaperData,
    ErrorResponse,
)

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'ArXiv Paper Retrieval API'
    }), 200


@api_bp.route('/papers/search', methods=['POST'])
def search_papers():
    """
    Search for papers on ArXiv.

    Request body:
    {
        "user_prompt": "papers about machine learning in biology",
        "llm_provider": "lmstudio",  # optional, default: "lmstudio"
        "model_id": "qwen/qwen3-32b"  # optional, uses config.py default if not provided
    }
    """
    try:
        req = PaperSearchRequest.model_validate(request.get_json() or {})
    except ValidationError as e:
        return jsonify(ErrorResponse(error='Invalid request body', details=str(e)).model_dump()), 400

    try:
        result = asyncio.run(
            run_arxiv_agent(
                user_prompt=req.user_prompt,
                llm_provider=req.llm_provider,
                model_id=req.model_id or LLM_MODEL,
            )
        )
    except Exception as e:
        return jsonify(ErrorResponse(error=str(e)).model_dump()), 400

    try:
        papers_df = result.get('papers_df')
        papers = [
            PaperData(
                arxiv_id=row.get('arxiv_id', ''),
                title=row.get('title', ''),
                summary=row.get('summary', ''),
                published=str(row.get('published', '')),
                updated=str(row.get('updated', '')),
                authors=row.get('authors', []),
                categories=row.get('categories', []),
                primary_category=row.get('primary_category', ''),
                doi=row.get('doi') or None,
                journal_ref=row.get('journal_ref') or None,
                comment=row.get('comment') or None,
            )
            for _, row in papers_df.iterrows()
        ]
        response = PaperSearchResponse(
            generated_query=result.get('arxiv_query', ''),
            num_papers=len(papers),
            papers=papers,
        )
        return jsonify(response.model_dump()), 200

    except Exception as e:
        return jsonify(ErrorResponse(error='Unexpected error', details=str(e)).model_dump()), 500
