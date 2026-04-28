import re

def clean_text(text):
    # lowercase
    text = text.lower()
    
    # usunięcie znaków specjalnych (zostawiamy litery i cyfry)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # usunięcie nadmiarowych spacji
    text = re.sub(r"\s+", " ", text).strip()
    
    return text