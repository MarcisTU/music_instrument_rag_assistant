import os
from datetime import datetime
from typing import Optional, List

from sqlmodel import SQLModel, Field, Column, Relationship
import sqlalchemy.dialects.postgresql as pg
from sqlalchemy import Text
from pgvector.sqlalchemy import Vector


class Product(SQLModel, table=True):
    __tablename__ = "products"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    url: str
    category_name: str
    description: str
    price: float
    store_name: str
    in_stock: bool
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            pg.TIMESTAMP,
            nullable=False,
            default=datetime.now(),
        )
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            pg.TIMESTAMP,
            nullable=False,
            default=datetime.now(),
            onupdate=datetime.now(),
        )
    )

    ### Relationships
    embedding: "ProductEmbedding" = Relationship(back_populates="product", cascade_delete=True)
    reviews: List["Review"] = Relationship(back_populates="product", cascade_delete=True)

    def __repr__(self):
        return f"<Product {self.name}>"


class ProductEmbedding(SQLModel, table=True):
    __tablename__ = "product_embeddings"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    structured_text: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    generated_description: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    structured_embedding: list[float] = Field(
        sa_column=Column(Vector(int(os.getenv("EMBEDDING_DIM"))), nullable=False)
    )
    generated_description_embedding: list[float] = Field(
        sa_column=Column(Vector(int(os.getenv("EMBEDDING_DIM"))), nullable=False)
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            pg.TIMESTAMP,
            nullable=False,
            default=datetime.now(),
        )
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            pg.TIMESTAMP,
            nullable=False,
            default=datetime.now(),
            onupdate=datetime.now(),
        )
    )

    product_id: int = Field(foreign_key="products.id", ondelete="CASCADE")

    ### Relationships
    product: "Product" = Relationship(back_populates="embedding")

    def __repr__(self):
        return f"<ProductEmbedding {self.name}>"


class Review(SQLModel, table=True):
    __tablename__ = "reviews"

    id: int | None = Field(default=None, primary_key=True)
    author: str
    text: str
    title: str
    rating: float
    votes_up: int
    votes_down: int
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            pg.TIMESTAMP,
            nullable=False,
            default=datetime.now(),
        )
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            pg.TIMESTAMP,
            nullable=False,
            default=datetime.now(),
            onupdate=datetime.now(),
        )
    )

    product_id: int = Field(foreign_key="products.id", ondelete="CASCADE")

    ### Relationships
    product: "Product" = Relationship(back_populates="reviews")

    def __repr__(self):
        return f"<Review {self.name}>"


class Task(SQLModel, table=True):
    __tablename__ = "tasks"

    id: int | None = Field(default=None, primary_key=True)
    task_uuid: str = Field(sa_column_kwargs={"unique": True, "index": True})
    status: str
    callback_url: Optional[str] = Field(default=None)
    user_query: str
    llm_result_text: Optional[str] = Field(default=None)
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            pg.TIMESTAMP,
            nullable=False,
            default=datetime.now(),
        )
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            pg.TIMESTAMP,
            nullable=False,
            default=datetime.now(),
            onupdate=datetime.now(),
        )
    )

    def __repr__(self):
        return f"<Task {self.id}>"
