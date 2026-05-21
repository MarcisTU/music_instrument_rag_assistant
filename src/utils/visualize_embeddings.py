import asyncio
from pathlib import Path
import numpy as np
import pandas as pd
import pacmap
import plotly.express as px
from loguru import logger

from src.db.service import ProductService


async def generate_embedding_map():
    logger.info("Fetching embeddings and categories from database...")

    records = await ProductService.get_all_embeddings()

    # Choose which embedding type to visualize structured/generated
    embedding_type = "generated_embedding"   # or - structured_embedding

    total_records = len(records)
    if total_records == 0:
        logger.error("No embeddings found in the database. Run your data ingest script first!")
        return

    logger.info(f"Successfully retrieved {total_records} embeddings.")

    names = [r["name"] for r in records]
    categories = [r["category_name"] for r in records]
    embeddings_matrix = np.array([r[embedding_type] for r in records], dtype=np.float32)

    logger.info(f"Matrix shape for reduction: {embeddings_matrix.shape}. Embedding type: {embedding_type}")
    logger.info("Initializing PaCMAP and fitting data (this may take a moment)...")

    n_neighbors = min(15, total_records - 1) if total_records <= 15 else None

    embedding_projector = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=42
    )

    components_2d = embedding_projector.fit_transform(embeddings_matrix, init="pca")
    logger.info("Dimensionality reduction completed successfully.")

    logger.info("Structuring data for visualization mapping...")
    df = pd.DataFrame({
        "x": components_2d[:, 0],
        "y": components_2d[:, 1],
        "product_name": names,
        "category": categories,
        "size_marker": 4
    })

    logger.info("Generating Plotly interactive engine graphics...")
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="category",
        hover_data={
            "product_name": True,
            "category": True,
            "x": False,
            "y": False,
            "size_marker": False
        },
        size="size_marker",
        width=1800,
        height=1000,
        title="<b>2D Visual Projection of Product Semantics via PaCMAP</b>"
    )

    fig.update_traces(
        marker=dict(opacity=0.8, line=dict(width=0.5, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        title_font_size=24,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=150),

        # Move legend to the bottom to give the main plot maximum horizontal space
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,  # Placed cleanly underneath the X-axis
            xanchor="center",
            x=0.5,  # Centered
            itemsizing="constant",
            traceorder="normal",
            title_text="<b>Product Categories:</b>"
        ),

        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    SCRIPT_DIR = Path(__file__).resolve().parent
    output_dir = SCRIPT_DIR.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "product_embeddings_map.html"

    logger.info(f"Writing static HTML artifact package to: {output_path}")
    fig.write_html(str(output_path))
    logger.info("Done! You can now open this HTML file directly in any browser on your computer.")


if __name__ == "__main__":
    asyncio.run(generate_embedding_map())