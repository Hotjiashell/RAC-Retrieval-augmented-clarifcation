

def extract_response(text, skip_special_tokens=True, chat=True):
    """
    Utility to extract clarifying question from model output
    """
    if chat:
        if skip_special_tokens:
            try:
                return (
                    text.split("assistant")[1]
                    .replace("\n", "")
                    .strip()
                )
            except:
                return text.strip()

        else:
            try:
                return (
                    text.split("assistant<|end_header_id|>")[1].replace(
                        "\n", "").split("<|eot_id|>")[0].strip()
                )
            except:
                return text.strip()
    else:
        return text.split("Clarifying Question:")[1].replace("\n", "").split("<|end_of_text|>")[0].strip()
