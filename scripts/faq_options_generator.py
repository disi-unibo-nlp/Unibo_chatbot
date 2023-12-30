from tqdm import tqdm
import json
import argparse
import random
import string
import os
from pathlib import Path

from parse_faqs import (
    preprocess_data,
    is_english
)

def __create_dpo(faq_data_section : dict):

    ds = {
        'prompt' : [],
        'chosen' : [],
        'rejected' : []
    }

    for q_ids, q in enumerate(faq_data_section['Question']):
        rej_set = list(set(range(len(faq_data_section['Question']))) - set([q_ids]))
        rej_ids = random.choice(rej_set)

        ds['prompt'].append(q)
        ds['chosen'].append(faq_data_section['Answer'][q_ids])
        ds['rejected'].append(faq_data_section['Answer'][rej_ids])

    return ds


def create_datasets(faqs_directory : str, 
                    out_path : str = 'data/unibo_faqs/',
                    include_langs : bool = False):


    # load each section's faqs separately
    dir_path = Path(faqs_directory)

    # Load all faqs
    json_files = dir_path.rglob("*.json")
    # Convert Path objects to string representation
    json_file_paths = [str(file_path) for file_path in json_files]

    dpo_dataset = {
        'prompt' : [],
        'chosen' : [],
        'rejected' : []
    }

    sft_dataset = {
        'Question' : [],
        'Answer' :  []
    }

    for faq_file in tqdm(json_file_paths):

        # .aspx not supported - no data insied
        if 'aspx' in faq_file: continue

        with open(faq_file, 'r') as f_in:
            _data = json.load(f_in)

        # Remove special chars and out-of-context information
        _data = preprocess_data(_data)

        # Update SFT dataset
        for key in sft_dataset.keys():
            for el in _data[key]:
                sft_dataset[key].append(el)

        # Create DPO instances
        dpo_sub_dataset = __create_dpo(_data)
        for key in dpo_dataset.keys():
            for el in dpo_sub_dataset[key]:
                dpo_dataset[key].append(el)

    
    if include_langs:
        # Divide instances by language
        print("Dividing by language ...")

        lang_dpo_dataset = {
            'it' : {'prompt' : [], 'chosen' : [], 'rejected' : []},
            'en' : {'prompt' : [], 'chosen' : [], 'rejected' : []}
        }

        data = list(zip(dpo_dataset['prompt'], dpo_dataset['chosen'], dpo_dataset['rejected']))
        for ids in tqdm(range(len(data)), desc='Parsing DPO dataset'):
            p, c, r = data[ids]
            lang = 'en' if is_english(p) else 'it'

            lang_dpo_dataset[lang]['prompt'].append(p)
            lang_dpo_dataset[lang]['chosen'].append(c)
            lang_dpo_dataset[lang]['rejected'].append(r)


        lang_sft_dataset = {
            'it' : {'Question' : [], 'Answer' : []},
            'en' : {'Question' : [], 'Answer' : []}
        }
        data = list(zip(sft_dataset['Question'], sft_dataset['Answer']))
        for ids in tqdm(range(len(data)), desc='Parsing SFT dataset'):
            q,a = data[ids]
            lang = 'en' if is_english(q) else 'it'

            lang_sft_dataset[lang]['Question'].append(q)
            lang_sft_dataset[lang]['Answer'].append(a)

        sft_dataset = lang_sft_dataset
        dpo_dataset = lang_dpo_dataset

        sft_out_path = os.path.join(out_path, 'faq_sft_lang_dataset.json')
        dpo_out_path = os.path.join(out_path, 'faq_dpo_lang_dataset.json')

    else:
        sft_out_path = os.path.join(out_path, 'faq_sft_dataset.json')
        dpo_out_path = os.path.join(out_path, 'faq_dpo_dataset.json')

    # Save datasets
    with open(sft_out_path, 'w') as f_out:
        json.dump(sft_dataset, f_out, ensure_ascii=False)

    with open(dpo_out_path, 'w') as f_out:
        json.dump(dpo_dataset, f_out, ensure_ascii=False)

        

if __name__ == "__main__":


    parser = argparse.ArgumentParser(prog='FAQ-Parser',
                                    description='Find and parse FAQs.')

    
    parser.add_argument('--faq_path', type=str, help='Base path for searching or loading FAQ files.')
    parser.add_argument('--out_path', type=str, help='Path for storing results.')
    parser.add_argument('--group_by_lang', action='store_true', default=False, help="Divide FAQs by language (Italian/Enlish).")
    args = parser.parse_args()



    create_datasets(faqs_directory=args.faq_path, out_path=args.out_path, include_langs=args.group_by_lang)


    



