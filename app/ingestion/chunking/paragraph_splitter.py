import logfire

def chunk_text(text:str, chunk_size=1500)-> List:
    #SEMANTIC SPLITIING INTO CHUNKS
    with logfire.span("Text chunking", len(text)):
        if not text:
            return []
        paragraph = text.split("/n/n")
        chunks=[]
        current_chunk=""
        for p in paragraph:
            if len(current_chunk)+len(p)<chunk_size:
                current_chunk+=p+"/n/n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk=p+"/n/n"
            
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if chunks.strip()]
        logfire("Chunking is Successful")
        return valid_chunks
    