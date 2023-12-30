# ----------------------------------------------------
# >> Zero-shot (no RAG) prompts

IT_PROMPT = """
### Istruzioni:
Rispondi alla seguente domanda riguardo i regolamenti, il contenuto dei corsi
e gli insegnamenti dell'Università di Bologna. Segui attentamente le regole seguenti:
- Le opzioni sono riportate in un elenco e sono identificate da una lettera minuscola (e.g. 'a', 'b', 'c', 'd', ..) seguita da una parentesi ')';
- Selezione l'informazione tra le opzioni proposte che ritieni essere più corretta; 
- Rispondi indicando solamente la lettera dell'opzione corrispondente.

<Esempio>
Ecco un esmepio di come formattare la risposta. L'opzione corretta
per la seguente domanda è la lettera 'c':

### Domanda:
Quando è nato Paul McCartney?

### Opzioni
\t a) 1990
\t b) 1200
\t c) 1942
\t d) 2010

### Risposta
(c)
<Fine Esempio/>

Di seguito somo riportate la domanda e le possibili opzioni.

### Domanda:
{}

### Opzioni:
{}

### Risposta:

"""

EN_PROMPT = """
### Instructions:
Answer the following question regarding the regulations, course content, and teachings of the University of Bologna. Follow the rules below:
- Options are listed in a bullet-pointed list and are identified by a lowercase letter (e.g., 'a', 'b', 'c', 'd', ..) followed by a parenthesis ')';
- Select the information among the provided options that you believe is most correct;
- Respond by indicating only the letter of the corresponding option.

<Example>
Here is an example of how to format the answer. The correct option for the following question is the letter 'c':

### Question:
When was Paul McCartney born?

### Options
\t a) 1990
\t b) 1200
\t c) 1942
\t d) 2010

### Answer
(c)
<End of Example/>

Below are the question and possible options.

### Question:
{}

### Options:
{}

### Answer:
"""


ZERO_SHOT_PROMPTS_BY_LANG = {
    'it' : IT_PROMPT,
    'en' : EN_PROMPT
}

# ----------------------------------------------------
# >> Prompts for Training
IT_TRIN_PROMPT = """
### Istruzioni:
Rispondi alla seguente domanda riguardo i regolamenti, il contenuto dei corsi
e gli insegnamenti dell'Università di Bologna. Rispondi in italiano ed in modo
corretto.

### Domanda:
{}

### Risposta:
{}
"""


EN_TRIN_PROMPT = """
### Istruzioni:
Answer the following question regarding the regulations, course content, and teachings of the University of Bologna. 
Reply in english and provide faithful answers.

### Question:
{}

### Answer:
{}
"""


TRAINING_PROMPTS = {
    'it' : IT_TRIN_PROMPT,
    'en' : EN_TRIN_PROMPT
}


# ----------------------------------------------------
## >> Prompts for RAG

qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, \
answer the query asking about citations over different topics.
Please provide your answer in the form of a structured JSON format containing \
a list of authors as the citations. Some examples are given below.

{few_shot_examples}

Query: {query_str}
Answer: \
"""