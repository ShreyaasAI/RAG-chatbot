import os
import sys
import  uuid
import  logfire
import json

from app.config import Settings
from app.services.retrieval.gemini_embeddings import embeded_texts, get_embedding_dim

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.ingestion.loaders.pdf_loader import parse_pdf
from app.ingestion.loaders.html_loader import parse_html_file

from app.ingestion.loaders.pptx_loader import parse_office
from app.ingestion.loaders.text_loader import text_parser
settings = Settings()
qdrant_client = QdrantClient(
    url= settings.QDRANT_URL,
    api_key= settings.QDRANT_API
)
embedding_dim = get_embedding_dim()
PROCESSED_DATA_DIR = "processed_data"
logfire.configure(service_name="Ingestion-Service")

def save_process_locally(data:dict, source_type:str, filename:str)->str:
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    destination = os.path.join(folder, f"{filename}.json")
    with open(destination,"w",encoding="utf-8")as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    return destination
def process_file(file_path: str, filename, source_type):
    try:
        logfire.info("Processing file", file_path=file_path)
        # check if file is pdf or html
        extension = file_path.lower().rsplit(".",1)[-1]
        if extension== 'pdf':
            parsed_data = parse_pdf(file_path)
        elif extension in ('htm'or 'html'):
            parsed_data = parse_html_file(file_path)
        elif extension in 'docx':
            parsed_data = parse_office(file_path)
        elif extension in 'pptx':
            parsed_data = parse_office(file_path)
        
        elif extension == 'txt':
            parsed_data = text_parser(file_path)
        else:
            logfire.warning(f"Skipping file scanning {filename}")
            return
        if not parsed_data or parsed_data.strip():
            
            logfire.warning("No data found in file", file_path=file_path)
            return

        #Now chunking
        chunks  = chunk_text(parsed_data)
        if not chunks:
            
            logfire.warning("No chunks found in file", file_path=file_path)
            return

        meta_data = {
            "source": source_type,
            "filename": filename,
            "file_path": file_path
        }
        local_data=save_process_locally(meta_data, source_type, filename)
        logfire.info("Local data saved", local_data)
        # generate embeddings
        with logfire.span("Vectorizing and indexing"):
            embeddings= embeded_texts(chunks)
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": text,
                        "meta_data": meta_data
                    }
                )
            for text, embedding in zip[tuple](chunks, embeddings)
            ]
        
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points
        )
    except Exception as e:
        logfire.error(f"Failed to process{filename}:{e}")
    
    
def process_directory(dir_path:str, source_type: str):
    with logfire.span("Scanning Directory",path={dir_path}, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,f))]
        logfire.info(f"Found {len(files)}files in {dir_path}")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)

def universal_ingestion(base_dir:str, explicit_source_type:str , wipe:bool= False):
    with logfire.span("Universal Ingestion Started", base_directory =base_dir):
        
        collection_name = settings.QDRANT_COLLECTION
        
        if not  qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE
                )
            )
            logfire.info(f"Created Collection{collection_name}")

        subdir = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]
        if not subdir:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(basedir).lower())
                source_type = (
                    "true" if "true" in base_name 
                else "noisy" if "noisy" in base_name
                else "general"
                )
                logfire.info("No base found so made one",source_type)
            process_directory(base_dir, source_type)
            logfire.info("Processed directory", path=base_dir)
        else:
            for subd in subdir:
                source_type = ( "true" if "true" in subd.lower()
                else "noisy" if "noisy" in subd.lower() else"general")
            process_directory(os.path.join(base_dir,subdir),source_type)
    # generate ids
    

    

    logfire.info("File processed and saved to vector db")
if __name__ == "__main__":
    clean_args=[a for a in sys.argv if a!="--wipe"]
    target_dir = clean_args[1] if len(clean_args)>1 else "DATA"
    source_value = clean_args[2] if len(clean_args)>2 else None
    if not os.path.exists(target_dir):
        print(f"Error path {target_dir} doesnt exist")
        sys.exit(1)
    universal_ingestion(target_dir,source_value)