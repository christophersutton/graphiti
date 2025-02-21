import spacy

# Load the spaCy English model
# Note: For better accuracy, consider using:
# - en_core_web_md (medium model)
# - en_core_web_lg (large model)
# - en_core_web_trf (transformer-based model)
nlp = spacy.load("en_core_web_lg")

def spacy_extract_entities(text: str) -> list[dict]:
    """
    Extract entities from text using spaCy.
    Returns a list of dictionaries, each with:
      - "entity": the extracted entity text (str)
      - "type": the spaCy entity label (str)
    
    Note: DATE and TIME entities are excluded from the results.
    Also includes significant noun phrases based on linguistic patterns.
    """
    doc = nlp(text)
    entities = []
    date_related_labels = ["DATE", "TIME"]
    
    # Get named entities
    for ent in doc.ents:
        
        # Skip any date/time related entities
        if ent.label_ not in date_related_labels:
            entities.append({"entity": ent.text.strip(), "type": ent.label_})
    
    # Add significant noun phrases based on linguistic patterns
    seen_texts = {ent["entity"].lower() for ent in entities}
    
    for chunk in doc.noun_chunks:
        # Skip if already caught as a named entity
        if chunk.text.strip().lower() in seen_texts:
            continue
            
        # Analyze the chunk's structure
        chunk_root = chunk.root
        
        # Look for compound noun phrases that likely represent roles/titles
        # These typically have compound or amod dependencies
        compounds = [token for token in chunk_root.children 
                    if token.dep_ in ('compound', 'amod')]
        
        # If we have a compound structure and the root is a noun
        if compounds and chunk_root.pos_ == 'NOUN':
            # Get the full span including all relevant modifiers
            start_idx = min(t.i for t in compounds + [chunk_root])
            end_idx = max(t.i for t in compounds + [chunk_root])
            span = doc[start_idx:end_idx + 1]
            
            # Only include if it's a multi-word phrase
            if len(span) > 1:
                # Determine type based on linguistic properties
                entity_type = "ROLE" if chunk_root.lemma_ in nlp.vocab else "NOUN_CHUNK"
                
                entities.append({
                    "entity": span.text.strip(),
                    "type": entity_type
                })
                seen_texts.add(span.text.strip().lower())
    
    # Debug print final results
    print(f"DEBUG - Final entities after filtering: {entities}")
    return entities

# Mapping of spaCy entity types to friendly names.
SPACY_ENTITY_FRIENDLY_MAP = {
    "PERSON": "Person",
    "NORP": "Nationality/Religious/Political Group",
    "FAC": "Facility",
    "ORG": "Organization",
    "GPE": "Geopolitical Entity",
    "LOC": "Location",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "WORK_OF_ART": "Work of Art",
    "LAW": "Law",
    "LANGUAGE": "Language",
    "DATE": "Date",
    "TIME": "Time",
    "PERCENT": "Percent",
    "MONEY": "Money",
    "QUANTITY": "Quantity",
    "ORDINAL": "Ordinal",
    "CARDINAL": "Cardinal Number"
}