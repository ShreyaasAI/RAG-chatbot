import logfire
def text_parser(filepath:str)->str:
    try:
        with open(filepath, "r", encode = "UTF-8", errors = 'ignore') as f:
            
            return f.read()

    except Exception as e:
        logfire.error(f"Text parsing failed for {file_path}")
        raise e 