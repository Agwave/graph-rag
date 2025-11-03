def trunk_by_paragraph(text: str) -> list[str]:
    texts = text.split("\n\n")
    return [t for t in texts if t]
