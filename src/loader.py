from pathlib import Path
from pypdf import PdfReader


def  cargar_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def cargar_pdf(path: Path) -> str:
    lector = PdfReader(path)

    pages = []

    for page in lector.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def cargar_document(path: Path) -> str:

    extension = path.suffix.lower()

    if extension == ".txt":
        return cargar_txt(path)

    if extension == ".pdf":
        return cargar_pdf(path)

    raise ValueError(
        f"Formato no soportado: {extension}"
    )


def cargar_documents(directory: str):

    folder = Path(directory)

    documents = []

    for path in folder.iterdir():

        if path.suffix.lower() in [".txt", ".pdf"]:

            content = cargar_document(path)

            documents.append({
                "filename": path.name,
                "content": content
            })

    return documents