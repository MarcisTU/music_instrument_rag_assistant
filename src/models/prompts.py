from typing import List

from pydantic import BaseModel, Field


### Prompt definitions
STRUCTURED_TEXT_TEMPLATE = """
Product Name: {name}
Price: {price}
Availability: {availability}
Features and Specifications:
{bullet_points}
"""

LLM_DESCRIPTION_TEMPLATE = """
<product_context>
    <name>{name}</name>
    <category>{category}</category>
    <description>{description}</description>
</product_context>
"""

LLM_PRODUCT_ITEM_TEMPLATE = """<product_item>
    <id>{id}</id>
    <name>{name}</name>
    <description>{description}</description>
</product_item>"""


# Used for removing non-technical related information
LLM_USER_QUERY_TRANSFORM_TEMPLATE = """
<user_input>{text}</user_input>

<rules>
    1. Extract only physical parameters, specifications, material types, and instrument categories.
    2. Completely ignore user intent, pronouns, conversational context, or questions (e.g., remove "I am looking for", "Can you find me").
    3. Output only the plain text paragraph of product features.
</rules>

<example>
    <input>"Hey I am a beginner looking for an affordable 6 string electric guitar with a maple neck and dual humbuckers for playing metal"</input>
    <output>"6-string electric guitar beginner-friendly maple neck dual humbucker pickups metal music style</output>
</example>
"""

LLM_PRODUCT_SUGGESTION_TEMPLATE = """
<context>
{context}
</context>

<user_query>
{user_query}
</user_query>
"""

### System prompts
LLM_SYS_PROMPT_USER_QUERY_NORMALIZATION = """
<task>Your task is to convert a user query into a clean, dense paragraph of standalone product characteristics.</task>
"""


### Structured output definitions
class AiWarmup(BaseModel):
    joke: str = Field()

class AiProductDescription(BaseModel):
    description: str = Field(description="Concise informative product description text")


class AiUserQuery(BaseModel):
    product_description: str = Field(description="Informative product description of user query")


### Structured output definitions
class AiProductResults(BaseModel):
    product_ids: List[int] = Field(
        min_length=5,
        max_length=5,
        description="LLM suggested product id's"
    )


if __name__ == "__main__":
    context = "".join([
        f"{LLM_PRODUCT_ITEM_TEMPLATE.format(name=doc['name'], id=doc['id'], description=doc['description'])}\n"
        for doc in [{"name": 'guitar', 'description': 'electric guitar with 6strings', 'id': 19},
                    {"name": 'guitar 2', 'description': 'electric guitar with 8 strings', 'id': 2}]
    ])
    print(context)
