import os
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone


class ProductRead(BaseModel):
    id: int
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    category_name: str = Field(..., min_length=1)
    description: str = Field(default="")
    price: float = Field(..., ge=0.0)
    store_name: str = Field(..., min_length=1)
    in_stock: bool = Field(default=False)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)   # allows to validate SQLModels from DB directly


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    category_name: str = Field(..., min_length=1)
    description: str = Field(default="")
    price: float = Field(..., ge=0.0)
    store_name: str = Field(..., min_length=1)
    in_stock: bool = Field(default=False)


class Review(BaseModel):
    id: int
    product_id: int = Field(default=-1)
    author: str = Field(default="Anonymous", min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    rating: float = Field(..., ge=0.0, le=5.0)
    votes_up: int = Field(default=0, ge=0)
    votes_down: int = Field(default=0, ge=0)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ReviewCreate(BaseModel):
    product_id: int = Field(default=-1)
    author: str = Field(default="Anonymous", min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    rating: float = Field(..., ge=0.0, le=5.0)
    votes_up: int = Field(default=0, ge=0)
    votes_down: int = Field(default=0, ge=0)


class ProductEmbeddingCreate(BaseModel):
    product_id: int = Field(default=-1)
    name: str = Field(..., min_length=1)
    structured_text: Optional[str] = None
    generated_description: Optional[str] = None
    structured_embedding: np.ndarray = Field(default_factory=lambda: np.zeros(int(os.getenv("EMBEDDING_DIM"))))
    generated_description_embedding: np.ndarray = Field(default_factory=lambda: np.zeros(int(os.getenv("EMBEDDING_DIM"))))

    class Config:
        arbitrary_types_allowed = True


class TaskUpdate(BaseModel):
    # This configuration makes it easy to work with ORM data structures seamlessly
    model_config = ConfigDict(from_attributes=True)

    status: Optional[str] = None
    llm_result_text: Optional[str] = None


class WorkerStatus(BaseModel):
    worker_id: str
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
