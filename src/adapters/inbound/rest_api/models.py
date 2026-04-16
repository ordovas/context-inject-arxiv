"""
Data models for REST API requests and responses.
"""
from pydantic import BaseModel
from typing import Optional, List


class PaperSearchRequest(BaseModel):
    """Request model for paper search endpoint."""
    
    user_prompt: str  # Natural language query
    llm_provider: str = "lmstudio"  # Optional: LLM provider
    model_id: Optional[str] = None  # Optional: Model identifier
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_prompt": "papers about machine learning in biology",
                "llm_provider": "lmstudio"
            }
        }


class PaperData(BaseModel):
    """Individual paper metadata."""
    
    arxiv_id: str
    title: str
    summary: str
    published: str
    updated: str
    authors: List[str]
    categories: List[str]
    primary_category: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comment: Optional[str] = None


class PaperSearchResponse(BaseModel):
    """Response model for paper search."""
    
    success: bool = True
    generated_query: str
    num_papers: int
    papers: List[PaperData]
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "generated_query": "cat:cs.LG AND all:biology",
                "num_papers": 2,
                "papers": [
                    {
                        "arxiv_id": "2301.12345",
                        "title": "Deep Learning for Biology",
                        "summary": "...",
                        "published": "2023-01-15",
                        "updated": "2023-01-20",
                        "authors": ["Alice", "Bob"],
                        "categories": ["cs.LG", "q-bio"],
                        "primary_category": "cs.LG"
                    }
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    success: bool = False
    error: str
    details: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Query generation failed",
                "details": "Error details here"
