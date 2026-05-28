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
<example>
    Here is an example of how you should analyze the product context and user query to identify the best matches.
    <context>
        <product_item>
            <id>1</id>
            <name>Solar Guitars A1.6FR-27</name>
            <description>6-string electric guitar built for progressive metal. Features a mahogany body, maple neck, ebony fretboard, super jumbo stainless steel frets, and a Floyd Rose 1000 locking tremolo. Equipped with passive Duncan Solar humbuckers.</description>
        </product_item>
        <product_item>
            <id>2</id>
            <name>Ibanez RG5120M Prestige</name>
            <description>Made in Japan hard rock and prog workhorse. Ash top on African mahogany body, 5-piece maple/wenge neck, bird's eye maple fretboard, and stainless steel frets. Loaded with active Fishman Fluence Modern Humbuckers and a Lo-Pro Edge double locking tremolo bridge.</description>
        </product_item>
        <product_item>
            <id>3</id>
            <name>Charvel Pro-Mod San Dimas Style 1 HH FR E</name>
            <description>Hard rock classic with modern updates. Alder body, speed neck made of roasted maple, 22 jumbo nickel frets, Floyd Rose 1000 series double-locking tremolo. Pickups are passive Seymour Duncan JB and '59 humbuckers.</description>
        </product_item>
        <product_item>
            <id>4</id>
            <name>Strandberg Boden Prog NX 6</name>
            <description>Ergonomic headless guitar for progressive metal. Sassafras body, maple top, roasted maple neck, richlite fretboard, and Jescar stainless steel frets. Active Fishman Fluence Modern humbuckers. Features a Strandberg EGS Rev7 tremolo system (not a locking nut/Floyd Rose design).</description>
        </product_item>
        <product_item>
            <id>5</id>
            <name>Schecter Reaper-6 Custom FR</name>
            <description>Modern metal machine. Khaya mahogany body with quilted maple top, ultra-thin C multi-ply roasted maple neck, ebony board, and extra jumbo stainless steel frets. Features a Floyd Rose 1500 locking tremolo and active Schecter San Andreas high-output humbuckers.</description>
        </product_item>
    </context>
    
    <user_query>
        Modern electric guitar for progressive metal and hard rock, with a roasted maple neck, stainless steel frets, and active humbuckers like Fishman Fluence. I’d like models similar with floyd rose locking tremolo.
    </user_query>
    
    <reasoning>
    1. Extract Constraints from User Query:
       - Genre: Progressive metal / hard rock
       - Neck: Roasted maple
       - Frets: Stainless steel
       - Pickups: Active humbuckers (specifically likes Fishman Fluence)
       - Bridge: Floyd Rose locking tremolo
    
    2. Evaluate Context Products against Constraints:
       - 1 (Solar): Has stainless steel frets and Floyd Rose. Misses roasted maple neck and active pickups (has passive Duncan Solars). Partial Match.
       - 2 (Ibanez): Matches progressive metal/hard rock, stainless steel frets, active Fishman Fluence humbuckers, and features an Ibanez Lo-Pro Edge double-locking tremolo (Floyd Rose style). Misses roasted maple neck (standard maple/wenge). High Match.
       - 3 (Charvel): Matches hard rock, roasted maple neck, and Floyd Rose. Misses stainless steel frets (nickel) and active pickups (passive Duncans). Partial Match.
       - 4 (Strandberg): Matches prog metal, roasted maple neck, stainless steel frets, and active Fishman Fluence. Misses Floyd Rose locking tremolo (uses non-locking Strandberg tremolo system). High Match but misses hardware constraint.
       - 5 (Schecter): Matches metal, roasted maple neck, stainless steel frets, Floyd Rose locking tremolo (FR 1500), and active humbuckers (San Andreas). Almost perfect match across all key hardware preferences.
    
    3. Final Selection Determination:
       - 5 matches nearly every strict constraint including hardware.
       - 2 matches the exact pickup choice (Fishman Fluence), frets, genre, and locking tremolo type.
    <reasoning>
    
    <output>
        [5, 2]
    </output>
</example>

Now perform similar analysis for the new data. Follow the thought process of the example above.

<user_query>
{user_query}
</user_query>

<context>
{context}
</context>
"""

### System prompts
LLM_SYS_PROMPT_USER_QUERY_NORMALIZATION = """
<task>Your task is to convert a user query into a clean, dense paragraph of standalone product characteristics.</task>
"""

LLM_SYS_PROMPT_DESCRIPTION = """Generate a concise informative description summary about given product information."""

LLM_SYS_NSHOT_PROMPT = """
<task>
    Given similar retrieved products context and user query suggest the most relevant products by returning their id values.
</task>
<rules>
    Pay attention to specific attributes that the user query contains and prioritize products in context that contain all or most of them.
</rules>
"""


### Structured output definitions
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
