import logfire
from  PyPDF2 import PdfReader

def parse_pdf(filepath:str)-> str:
    with logfire("parse_pdf", filepath=filepath):
        try:
            reader = PdfReader(filepath)
            total_pages =len(reader)
            logfire.info(f"Total pages in pdf are {total_pages}")
            text_parts = []
            blank_pages = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text.strip():  # Check if the extracted text is empty or only whitespace
                    blank_pages.append(page_num + 1)  # Page numbers are usually 1-indexed
                text_parts.append(text)
            if blank_pages:
                
                logfire.warn(f"Blank pages found at page numbers: {blank_pages} retrying with pyplumber")
                try:
                    import pdfplumber
                    with pdfplumber.open(filepath) as pdf:
                        new_text_parts = [page.extract_text() for page in pdf.pages]
                        
                        if new_text_parts.split():
                            text_parts.append(new_text_parts)
                except Exception  as plumber_err:
                    logfire.error(f"Error while parsing pdf with pdfplumber {filepath} with error {str(plumber_err)}")
                    raise plumber_err
                    
            full_text = "/n".join(text_parts)
            if not full_text:
                logfire.warning(f"No text found from {file_path}, the file maybe purely images")
            else:
                logfire.info(f"Successfully parsed pdf {filepath} and got {len(full_text)} characters")
            return full_text
        except Exception as e:
            logfire.error(f"Error while parsing pdf {filepath} with error {str(e)}")
            raise e