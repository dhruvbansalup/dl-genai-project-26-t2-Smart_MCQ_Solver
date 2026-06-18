import re

def clean_text(text: str) -> str:
    
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # replace multi space to single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def format_mcq_data(df):
    df["prompt"] = df["prompt"].apply(clean_text)
    for opt in ["A","B","C","D","E"]:
        df[opt] = df[opt].apply(clean_text)

    if "answer" in df.columns:
        df["answer"] = df["answer"].apply(clean_text)
    return df