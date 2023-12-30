from peft import PeftModel
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
import json
from tqdm import tqdm
import numpy as np
from transformers import pipeline



prompts = {
    'it' : """
### Istruzioni:
Rispondi alla seguente domanda riguardo i regolamenti, il contenuto dei corsi
e gli insegnamenti dell'Università di Bologna. Segui attentamente le regole seguenti:
- Le opzioni sono riportate in un elenco e sono identificate da una lettera minuscola seguita e preceduta da una parentesi tonda;
- Selezione l'informazione tra le opzioni proposte che ritieni essere più corretta;
- Rispondi indicando SOLAMENTE la lettera dell'opzione corrispondente tra parentesi (for instance (correct_letter).

<Esempio>
Ecco un esmepio di come formattare la risposta:
### Domanda:
Quando è nato Paul McCartney?

Scegli una delle seguenti opzioni:
### Opzioni
\t a) 1990
\t b) 1200
\t c) 1942
\t d) 2010

### Risposta
(c)
<Fine Esempio>

Di seguito somo riportate la domanda e le possibili opzioni.

### Domanda:
{}

Scegli una delle seguenti opzioni:
### Opzioni:
{}

### Risposta:
""",
"en" : """
### Instructions:
Answer the following question regarding the regulations, course content, and teachings of the University of Bologna. Follow the rules below:
- Options are listed in a bullet-pointed list and are identified by a lowercase letter with a rounded parenthesis at the beggining and at the end;
- Select the information among the provided options that you believe is most correct;
- Respond by indicating ONLY the letter of the corresponding option.

<Example>
Here is an example of how to format the answer. The correct option for the following question is the letter 'c':

### Question:
When was Paul McCartney born?

Choose amnong the following options:
### Options
\t a) 1990
\t b) 1200
\t c) 1942
\t d) 2010

### Answer
(c)
<End of Example>

Below are the question and possible options.

### Question:
{}

Choose amnong the following options:
### Options:
{}

### Answer:
"""
}



# evaluation pipeline
def evaluation_loop(llm):
    results = {}

    for lang in ['it', 'en']:
        data = mc_lang_faq[lang]

        print("\n\n", "-"*20,'\n',lang)
        answers = []

        prompt = prompts[lang]
        tag = '# Risposta' if lang == 'it' else '# Answer'

        for quest, opt in tqdm(zip(data['Question'], data['Options']), total=len(data['Question'])):

            quest = quest.split('on:     __ \n on Instagram ')[0].strip()
            opt = opt.split('on:     __ \n on Instagram ')[0].strip()

            query = prompt.format(quest, opt)
            outputs = llm(query, do_sample=False, max_new_tokens=8)#, temperature=0.2, top_k=20, max_new_tokens=8, do_sample=True)
            
            gen_text = outputs[0]['generated_text'].split(tag)[-1]
            letter = gen_text.split(')')[0]
            if '(' in letter: letter = letter.split('(')[1]
            answers.append(letter.strip())

        res = [pred == label for pred, label in zip(answers, data['Answer'])]

        results[lang] = round(np.count_nonzero(res) / len(data['Question']) * 100, 4)

    return results



model_name = 'upstage/SOLAR-10.7B-Instruct-v1.0'

model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            device_map='cuda:0', 
                                            torch_dtype=torch.float16,
                                            cache_dir='cache/',
                                            attn_implementation = "flash_attention_2")

tokenizer = AutoTokenizer.from_pretrained(model_name)


llm = pipeline("text-generation", model, tokenizer=tokenizer)
res = evaluation_loop(llm)

print(res)

