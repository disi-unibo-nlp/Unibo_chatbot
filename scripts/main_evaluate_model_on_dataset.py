
import evaluate
import torch
import nltk
from tqdm import tqdm
import numpy as np
import json


nltk.download("punkt", quiet=True)
metric_rouge = evaluate.load("rouge")



class DataEvaluator():

    def __init__(self,
                 gen_pipeline):
        super(DataEvaluator, self).__init__()

        self.pipe = gen_pipeline

    def __compute_rouge_scores(self,
                               prediction,
                               label):

        # rougeLSum expects newline after each sentencexe
        decoded_preds = ["\n".join(nltk.sent_tokenize(prediction))]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label))]

        result = metric_rouge.compute(predictions=decoded_preds,
                                        references=decoded_labels,
                                        use_stemmer=True)

        result = {k: round(v * 100, 2) for k, v in result.items()}

        result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
            (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

        decoded_preds = [pred.replace("\n", " ") for pred in decoded_preds]
        result["gen_len"] = np.count_nonzero([pred != tokenizer.pad_token_id for pred in decoded_preds[0]])

        return result

    
    def __evaluation_generative_pipeline(self,
                                        pipe,
                                        dataset,
                                        answer_tag : str = '# Risposta:',
                                        question_tag : str = '# Domanda:',
                                        text_field : str ='text',
                                        label_field : str = 'answer',
                                        max_new_token : int = 32):

        overall_stats = {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'rougeLsum' : 0.0,
            'R' : 0.0,
            'gen_len' : 0.0
        }

        generated_answers = []

        for row in tqdm(dataset):
            input_prompt = row[text_field]
            true_answ = row[label_field]

            outputs = pipe(input_prompt,
                            max_new_tokens=max_new_token,
                            num_return_sequences=1,
                            do_sample=False,
                            pad_token_id=pipe.tokenizer.eos_token_id,
                            temperature=0.0)

            output = outputs[0]["generated_text"]
            prediction = output.split(answer_tag)[-1]

            results = self.__compute_rouge_scores(prediction, true_answ)
            for k in results.keys(): overall_stats[k] += results[k]

            generated_answers.append(prediction.strip())

        for k in overall_stats.keys():
            overall_stats[k] /= dataset.shape[0]

        return overall_stats, generated_answers


    def evaluate_dataset(self,
                         dataset,
                         max_new_tokens,
                         question_field,
                         amswer_field,
                         stats_out_file,
                         generated_samples_out_file):
        """
        Evaluate the input dataset
        """
        
        evaluation_config = {
            'pipe' : self.pipe,
            'dataset' : dataset,
            'text_field' : question_field,
            'label_field' : label_field,
            'max_new_tokens' : max_new_tokens
        }

        stats, gen_out = self.__evaluation_generative_pipeline(**evaluation_config)

        with open(stats_out_file, 'w') as f_out:
            json.dump(stats, f_out)

        with opem(generated_samples_out_file, "w") as f_out:
            f_out.write(f'GENERATED ANSWERS:\n\t - ' + [f'{i}. {answ}\n' for i,answ in enumerate(gen_out)].join('\t - '))

        






def formatting_prompts_func(examples,
                            prompt : str = '',
                            question_tag : str = 'Question',
                            answer_tag : str = 'Answer',
                            question_field : str = 'question',
                            answer_field : str = 'answer'):

    output_texts = []
    for ex_ids in range(len(examples[question_tag])):
        prompt = prompt.format(examples[question_field][ex_ids],
                               examples[answer_field][ex_ids])
        output_texts.append(prompt)

    return output_texts


instruction = """### Istruzioni:
Sei un membro della commissione valutatrice del concorso per l'accesso
agli uffici pubblici. Basandoti sulle tue conoscenze del Codice degli Appaloti (D.Lgs. 36/2023)
rispondi alla seguente domanda in italiano.

### Domanda:
{}

### Risposta:
"""

prompt_args = {
    'prompt' : instruction,
    'question_tag' : 'Domanda',
    'question_field' : 'Domanda',
    'answer_tag' : 'Risposta',
    'answer_field' : 'Risposta'
}

prompting_function = lambda x : formatting_prompts_func(x,**prompt_args)
evaluation_metric = lambda x : compute_metrics(x, f"# {prompt_args['answer_tag']}:")




pipe = pipeline("text-generation",
                model=model,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto")


split_dim = 10
prompted_q = ca_dataset[:split_dim]['Domanda'].apply(lambda x : instruction.format(x))
prompted_ds = pd.DataFrame.from_dict({'Domanda' : prompted_q, 'Risposta' : ca_dataset['Risposta'][:split_dim]})
prompted_ds = Dataset.from_pandas(prompted_ds)

metrics, gen_out = evaluation_generative_pipeline(pipe,
                                        prompted_ds,
                                        answer_tag='### Risposta:',
                                        question_tag='### Domanda:',
                                        text_field='Domanda',
                                        label_field='Risposta',
                                        max_new_token=256,
                                        print_results=False)                                        