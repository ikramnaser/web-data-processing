# Web Data Processing 

This project presents a method for extracting and validating structured knowledge from the output of a Large
Language Model. Specifically, the system utilizes Llama2 7B to generate raw text output and the BERT-based
model for Named Entity Recognition (NER) to extract relevant entities from the input and generated text.
The program’s functionality is structured into four main tasks:
1. Task 1: Retrieve the raw output text from the Llama model.
2. Task 2: Extract and list entities from both the input text and the raw text output.
3. Task 3: Extract an answer (either ”yes/no” or a Wikipedia entity link).
4. Task 4: Evaluate the correctness of the extracted answer.
