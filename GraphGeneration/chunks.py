from pydantic import Field ,BaseModel
from typing import List
class Chunks(BaseModel):
    """Chunks of text to provide to the user"""
    chunk: List[str] = Field(description="List of the chunks generated")