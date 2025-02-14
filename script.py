import re
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from llama_cpp import Llama


# Load BERT model and tokenizer for NER
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Llama model path
llama_model_path = r"C:\Users\ikry\models\llama-2-7b.Q2_K.gguf"
llm = Llama(model_path=llama_model_path, verbose=False)

# extract entities using BERT
def extract_entities(text):
    ner_results = ner_pipeline(text)
    entities = list(set(result["word"] for result in ner_results))
    return entities

# query the Llama model and extract raw text
def query_llama(question):
    output = llm(
        question,
        max_tokens=128,
        echo=True
    )
    return output["choices"][0]["text"]

# get Wikipedia link for an entity
def get_wikipedia_link(entity):
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": entity,
        "prop": "info",
        "inprop": "url"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id != "-1":
                return page_data.get("fullurl")
    except requests.exceptions.RequestException as e:
        print(f"Error querying Wikipedia for entity '{entity}': {e}")
    return None

# entity disambiguation using Wikidata
def disambiguate_entity(entity_name, context):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity_name
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("search", [])
        for result in results:
            description = result.get("description", "")
            if any(word.lower() in description.lower() for word in context.split()):
                return result["label"], result["description"]
    return entity_name, "No disambiguation found"

# entity linking
def link_entities(entities, context):
    linked_entities = []
    for entity in entities:
        disambiguated_name, description = disambiguate_entity(entity, context)
        link = get_wikipedia_link(disambiguated_name)
        if link:
            linked_entities.append((disambiguated_name, link, description))
    return linked_entities

# Function to extract the answer 
#def extract_answer(question, raw_text, linked_entities):
    #if question.lower().startswith(("is", "does", "are", "was", "were", "can", "should")):
       # if re.search(r"\b(yes|yeah|yep)\b", raw_text, re.IGNORECASE):
          #  return "yes"
        #elif re.search(r"\b(no|nope|not at all)\b", raw_text, re.IGNORECASE):
            #return "no"
    #if linked_entities:
        #return linked_entities[0][1]  # Return the Wikipedia link of the first matched entity
    #return "Answer not found"

# extract the answer (IMPROVED VERSION)
def extract_answer(question, raw_text, linked_entities):
    # Define the Yes/No classification function 
    def classify_yes_no(text):
        yes_patterns = [r"\b(yes|yeah|yep|correct|true|indeed|absolutely|definitely|of course)\b", r"it is", r"that's right", r"without a doubt"]
        no_patterns = [r"\b(no|nope|false|not at all|incorrect|never|absolutely not)\b", r"it is not", r"that's wrong", r"under no circumstances"]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in yes_patterns):
            return "yes"
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in no_patterns):
            return "no"
        return "Answer not found"
    
    if question.lower().startswith(("is", "does", "are", "was", "were", "can", "should")):
        return classify_yes_no(raw_text)
    
    # Check if raw_text contains any entities, and if so, return the entity name
    for entity in linked_entities:
        if entity[0] in raw_text:
            return entity[0]  
    
    return "Answer not found"



# fact-check the answer
def fact_check_answer(question, extracted_answer, linked_entities):
    if extracted_answer in ["yes", "no"]:
        return "correct" if linked_entities else "incorrect"
    for entity, url, description in linked_entities:
        if extracted_answer == url:
            try:
                response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity}")
                response.raise_for_status()
                summary = response.json().get("extract", "")
                if entity.lower() in summary.lower() and all(
                    word.lower() in summary.lower() for word in question.split()
                ):
                    return "correct"
            except requests.exceptions.RequestException as e:
                print(f"Error fetching evidence for '{entity}': {e}")
    return "incorrect"

# Combined function for all tasks
def process_question(question):
    input_entities = extract_entities(question)
    raw_text = query_llama(question)
    output_entities = extract_entities(raw_text)
    linked_entities = link_entities(output_entities, question)
    extracted_answer = extract_answer(question, raw_text, linked_entities)
    correctness = fact_check_answer(question, extracted_answer, linked_entities)
    result = {
        "Input (A)": question,
        "Entities in Input (A)": input_entities,
        "Raw Text (B)": raw_text,
        "Entities in Raw Text (B)": output_entities,
        "Linked Entities": linked_entities,
        "Extracted Answer": extracted_answer,
        "Correctness": correctness
    }
    return result

# Example questions
questions = [
    "Is Managua the capital of Nicaragua?",
    "Is it true that China is the country with most people in the world?",
    "The largest company in the world by revenue is Apple.",
    "Who is the director of Pulp Fiction?",
    "Is it true that the monarch of England is also the monarch of Canada?"
]

# Process the question
for i, q in enumerate(questions):
    print(f"Processing: question-{i+1:03d} {q}")
    result = process_question(q)
    print("Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
    print()
