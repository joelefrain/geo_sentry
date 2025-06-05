def to_sentence_format(text: str, mode: str = "lower") -> str:
    """
    Formatea un nombre de serie aplicando un tipo de conversión a la parte antes del paréntesis
    y conservando intacto el contenido entre paréntesis.

    Parámetros:
        text (str): El texto original.
        mode (str): Tipo de conversión aplicado a la parte antes del paréntesis. Opciones:
            - 'lower':        convierte todo a minúsculas (ej. "TURBIDEZ (NTU)" → "turbidez (NTU)")
            - 'sentence':     primera letra en minúscula, el resto igual (ej. "Presión (kPa)" → "presión (kPa)")
            - 'capitalize':   primera letra en mayúscula, el resto en minúsculas (ej. "Presión (kPa)" → "Presión (kPa)")
            - 'title':        tipo título (mayúscula inicial de cada palabra) (ej. "oxígeno disuelto (mg/L)" → "Oxígeno Disuelto (mg/L)")
            - 'original':     mantiene el texto tal como está (ej. "Presión (kPa)" → "Presión (kPa)")
           - 'decapitalize': convierte solo la primera letra a minúscula, dejando el resto intacto.

    Retorna:
        str: El texto formateado.

    Ejemplos:
        >>> to_sentence_format("Presión (kPa)", mode="lower")
        'presión (kPa)'

        >>> to_sentence_format("TURBIDEZ (NTU)", mode="sentence")
        'turbidez (NTU)'

        >>> to_sentence_format("oxígeno disuelto (mg/L)", mode="capitalize")
        'Oxígeno disuelto (mg/L)'

        >>> to_sentence_format("oxígeno disuelto (mg/L)", mode="title")
        'Oxígeno Disuelto (mg/L)'

        >>> to_sentence_format("pH in-situ (mg/L)", mode="original")
        'pH in-situ (mg/L)'
        
        >>> to_sentence_format("PH in-situ (mg/L)", mode="decapitalize")
        'pH in-situ (mg/L)'"""
    text = text.strip()
    
    if '(' in text:
        main = text.split('(')[0].strip()
        suffix = text[text.find('('):]
    else:
        main = text
        suffix = ''
    
    if mode == "lower":
        main = main.lower()
    elif mode == "sentence":
        main = main.lower()
    elif mode == "capitalize":
        main = main.capitalize()
    elif mode == "title":
        main = main.title()
    elif mode == "original":
        pass
    elif mode == "decapitalize":
        main = main[0].lower() + main[1:] if main else ''
    else:
        raise ValueError(f"Modo de conversión no válido: {mode}")
    
    return f"{main} {suffix}".strip()