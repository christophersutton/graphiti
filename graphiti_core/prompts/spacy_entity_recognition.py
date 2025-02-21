"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Protocol
from pydantic import BaseModel, Field

from graphiti_core.prompts.models import Message, PromptFunction

class SpacyEntity(BaseModel):
    entity: str = Field(..., description="The extracted entity text")
    type: str = Field(..., description="The type of the entity")

class SpacyEntityResponse(BaseModel):
    status: str = Field(..., description="Status of the extraction validation ('Good' or 'Bad')")
    revised_entities: list[SpacyEntity] = Field(default_factory=list, description="List of revised entities if status is 'Bad', empty list otherwise")

class Prompt(Protocol):
    v1: PromptFunction

class Versions(dict[str, PromptFunction]):
    pass

def spacy_entity_confirmation(context: dict) -> list[Message]:
    """
    Returns a prompt for the LLM to confirm or revise the entity extraction.
    
    The context dictionary should include:
      - "episode_content": The original text.
      - "spacy_entities": The list of entity dictionaries (each with "entity" and "type")
    
    The LLM is instructed to return a JSON object with the following structure:
      {"status": "<Good or Bad>", "revised_entities": <list of entities if status is Bad, else []>}
    """
    sys_prompt = (
        "You are an expert in named entity recognition. Below is a text excerpt and a list of entities "
        "extracted from it (with their types) using a fast rule-based algorithm. "
        "Evaluate whether the list is accurate and comprehensive.\n\n"
        "Guidelines for evaluation:\n"
        "1. Do NOT include temporal information like dates, times, or years as entities\n"
        "2. Focus on significant entities, concepts, and actors mentioned in the text\n"
        "3. Avoid including relationships or actions as entities\n"
        "4. Use full names and avoid abbreviations where possible\n\n"
        "Return a JSON object with the following format:\n"
        '{"status": "<Good or Bad>", "revised_entities": [<each entity as {"entity": ..., "type": ...}>]}\n'
        "If the list is accurate and complete, reply with status 'Good' and an empty revised_entities list. "
        "If there are errors or missing entities, reply with status 'Bad' and include a corrected list."
    )
    
    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>

<SPACY_ENTITIES>
{context['spacy_entities']}
</SPACY_ENTITIES>

Please evaluate the entity extraction as described.
    """
    
    return [
        Message(role="system", content=sys_prompt),
        Message(role="user", content=user_prompt)
    ]

versions: Versions = {
    'v1': spacy_entity_confirmation
}