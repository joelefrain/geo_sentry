from reportlab.platypus import Paragraph, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path
import toml

class NotesHandler:
    def __init__(self, style_name="default"):
        """
        Inicializa el manejador de notas con un estilo específico
        :param style_name: Nombre del archivo de estilo (sin extensión .toml)
        """
        self.styles = getSampleStyleSheet()
        self.style_name = style_name
        self.style_config = self._load_style_config()
    
    def _load_style_config(self):
        """Carga la configuración de estilo desde un archivo TOML"""
        config_dir = Path(__file__).parent / "data" / "notes"
        config_path = config_dir / f"{self.style_name}.toml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Style config not found: {config_path}")
        
        return toml.load(str(config_path))
    
    def create_notes(self, sections, title_style=None, list_style=None):
        """Crea bloques de notas con formato configurable"""
        elements = []
        
        for section in sections:
            section_title_style = section.get('title_style', title_style or {})
            title_element = self._create_notes_title(
                section['title'],
                **section_title_style
            )
            elements.append(title_element)
            
            section_format = section.get('format_type', 'numbered')
            section_list_style = section.get('style', list_style or {})
            format_config = self.style_config['formats'].get(section_format, {})
            content_element = self._create_notes_items(
                section['content'],
                format_type=section_format,
                format_config=format_config,
                **section_list_style
            )
            elements.append(content_element)
        
        return elements

    def _create_notes_title(self, title, **kwargs):
        """Crea el elemento de título con el estilo configurado"""
        title_style = self.styles["Normal"].clone(name="NotesTitle")
        title_config = self.style_config['title']
        
        for key, value in title_config.items():
            setattr(title_style, key, value)
        
        for key, value in kwargs.items():
            setattr(title_style, key, value)
        
        return Paragraph(title, title_style)

    def _create_list_items(self, notes, text_style, format_config):
        """Método base para crear elementos de lista con cualquier formato"""
        config = format_config.copy()
        
        # Asegurar que el tamaño de la fuente del bullet coincida
        if 'bulletFontSize' not in config:
            config['bulletFontSize'] = text_style.fontSize
        
        # Extraer leftIndent de la configuración y eliminarlo del diccionario
        left_indent = config.pop('leftIndent', 0)
        
        # Crear los items de la lista
        note_items = [ListItem(Paragraph(note, text_style)) for note in notes]
        return ListFlowable(note_items, leftIndent=left_indent, **config)

    def _create_notes_items(self, notes, format_type, format_config, **kwargs):
        """Crea los elementos de contenido con el estilo configurado"""
        text_style = self.styles["Normal"].clone(name="NoteText")
        content_config = self.style_config['content']
        
        # Apply content configuration from TOML
        for key, value in content_config.items():
            setattr(text_style, key, value)
        
        # Apply format-specific configuration
        for key, value in format_config.items():
            setattr(text_style, key, value)
        
        # Apply additional overrides
        for key, value in kwargs.items():
            setattr(text_style, key, value)
        
        if isinstance(notes, str):
            notes = [notes]
        
        # Ensure all items in notes are strings
        notes_str = []
        for note in notes:
            if isinstance(note, dict):
                # If it's a dictionary, convert it to a string representation
                # This handles cases where a dictionary is passed instead of a string
                note_str = str(note)
                notes_str.append(note_str)
            else:
                notes_str.append(str(note))
        
        # Unified logic for handling different formats
        if format_type == 'paragraph':
            return Paragraph('<br/>'.join(notes_str), text_style)
        elif format_type in ['bullet', 'numbered', 'alphabet']:
            return self._create_list_items(notes_str, text_style, format_config)
        else:
            raise ValueError(f"Unsupported format_type: {format_type}")
