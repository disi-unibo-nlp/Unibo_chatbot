import html2text
import os, os.path
import errno
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import random
import string
import langid

def safe_open_w(path):
    """ 
    Open "path" for writing, creating any parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')


def __shuffle_words(sentence):
    # Split the sentence into words
    words = sentence.split()

    # Shuffle the order of words
    random.shuffle(words)

    # Join the shuffled words back into a sentence
    shuffled_sentence = ' '.join(words)

    return shuffled_sentence


def create_multiple_option_ds(qa_dict : dict, 
                              num_options : int = 3):
    """
    Create negative options for each question in the input qa_dict.

    If len(qa_dict.Questions) < num_options --> shuffle other options
    """

    multiple_choice_qa = {'Question':[], 'Options':[],'Answer':[]}

    for q_id, q in enumerate(qa_dict['Question']):
        if q == '' or qa_dict['Answer'][q_id] == '': continue

        multiple_choice_qa['Question'].append(q)

        set_q = list(set(range(len(qa_dict['Question']))) - set([q_id]))
        
        if len(set_q) >= num_options:
            opt_ids = random.sample(set_q, num_options)            
        else:
            opt_ids = set_q
        
        options = [qa_dict['Answer'][_opt_ids] for _opt_ids in opt_ids]
        

        if len(set_q) < num_options:
            for _ in range(num_options - len(set_q)): 
                # Shuffle options to replace missing options
                rand_ids = random.choice(set_q)
                rand_answ = qa_dict['Answer'][rand_ids]
                options.append(__shuffle_words(rand_answ))

        multiple_choice_qa['Options'].append(options)

    # Create options with letters
    letters = string.ascii_lowercase
    assert len(letters) >= num_options
    letters = letters[:num_options+1]

    for q_id, q in enumerate(qa_dict['Question']):
        options = multiple_choice_qa['Options'][q_id]
        correct_pos = random.choice(list(range(num_options+1)))
        correct_letter = letters[correct_pos]

        iter_options = iter(options)
        options_str = ''

        for ids in range(num_options+1):
            if ids == correct_pos:
                multiple_choice_qa['Answer'].append(correct_letter)
                opt = qa_dict['Answer'][q_id]
            else: 
                opt = next(iter_options)
            
            let = letters[ids]
            options_str += f'\n\t {let}) {opt}'

        multiple_choice_qa['Options'][q_id] = options_str

    
    return multiple_choice_qa

                
def preprocess_data(qa_dict):

    res = {'Question':[],'Answer':[]}

    for _q,_a in zip(qa_dict['Question'], qa_dict['Answer']):
        parsed_dict = {'q': _q, 'a':_a}
        for key in parsed_dict:
            val = parsed_dict[key]
            # Handle last line
            end_boundaries = ['Follow Unibo','Segui Unibo', 'Segui il corso', '__ Invia ad un amico', 'idsite']
            for _eb in end_boundaries: 
                if _eb in val: parsed_dict[key] = '\n'.join(val.split(_eb)[:-1])
            # Handle first line
            start_boundaries = ['Domande frequenti']
            for _sb in start_boundaries: 
                if _sb in val: parsed_dict[key] = val.split(_sb)[1]
            
        if (q := parsed_dict['q']) != '' and (a := parsed_dict['a']) != '':
            
            # Remove spourious chars
            q = q.replace('__','').strip()
            a = a.replace('__','').strip()

            res['Question'].append(q)
            res['Answer'].append(a)

    return res

def is_english(sentence : str):
    lang, score = langid.classify(sentence)
    return lang == 'en'

def __divide_faq_by_lang(data : dict):
    keys = ['Question','Options','Answer']
    faq_by_lang = {
        'en' : {
            'Question' : [], 'Options' : [], 'Answer' : []
        }, 'it' : {
            'Question' : [], 'Options' : [], 'Answer' : []
        }
    }

    for q_ids, q in enumerate(tqdm(data['Question'])):
        lang = 'en' if is_english(q) else 'it'
        for key in keys:
            faq_by_lang[lang][key].append(data[key][q_ids])

    return faq_by_lang

def divide_faq_by_lang(faq_in_file: str, 
                       faq_out_file: str):

    data = {}
    with open(faq_in_file, 'r') as f_in:
        data = json.load(f_in) 

    faq_by_lang = __divide_faq_by_lang(data)

    n_it_faqs = len(faq_by_lang['it']['Question'])
    n_en_faqs = len(faq_by_lang['en']['Question'])

    print(f"[STATS] # FAQs by language:\n\t - Italian -> {n_it_faqs}\n\t - English -> {n_en_faqs}")

    with open(faq_out_file, 'w') as f_out:
        json.dump(faq_by_lang, f_out, ensure_ascii=False)


def load_faqs_into_ds(faqs_directory : str,
                      out_path : str,
                      create_options : bool = False,
                      num_options : int = 3):

    dir_path = Path(faqs_directory)

    # Load all faqs
    json_files = dir_path.rglob("*.json")
    # Convert Path objects to string representation
    json_file_paths = [str(file_path) for file_path in json_files]


    keys = ['Question', 'Answer', 'Options'] if create_options else ['Question', 'Answer', 'Options']
    faqs_unique = {key : [] for key in keys}


    for faq_file in tqdm(json_file_paths):

        # .aspx not supported - no data insied
        if 'aspx' in faq_file: continue

        with open(faq_file, 'r') as f_in:
            _data = json.load(f_in)

        # Remove special chars and out-of-context information
        _data = preprocess_data(_data)

        if create_options:
            _data = create_multiple_option_ds(qa_dict=_data, 
                                              num_options=num_options)

        for key in faqs_unique.keys():
            for el in _data[key]:
                faqs_unique[key].append(el)


    with open(out_path, 'w') as f_out:
        json.dump(faqs_unique, f_out, ensure_ascii=False)

    print(f"[STATS] Parsed: {len(faqs_unique['Question'])} FAQs")


def parse_faqs(base_path, 
               out_path : str = 'data/unibo_faqs'):

    ## Init parser
    parser = html2text.HTML2Text()
    parser.ignore_links = True # Ignore converting links from HTML

    out_base_path = out_path

    ## Get path to FAQ files  
    faq_paths = list(Path(base_path).rglob("*faq*"))
    faq_files = [str(f) for f in faq_paths if not f.is_dir()]

    ## Parse FAQ into dictionary

    for _faq in tqdm(faq_files):
        if '.pdf' not in _faq:
            with open(_faq,'r', errors='ignore') as f_in:
                qa = f_in.read()
                
            parsed_qa = parser.handle(qa)

            # Divide into questions and answers
            delimiter = '##' if '##' in parsed_qa else '**'
            delimiter = '\n\n' + delimiter

            qa_blocks = [_qa_block for _qa_block in parsed_qa.split(delimiter) if '?' in _qa_block]
            qa_pairs = [tuple(_qa) for _qa_pair in qa_blocks if len(_qa := _qa_pair.split('?')) > 1]

            # Post process data
            special_characters=['@','#','$','*','&','<<','>>']
            filter_out_tokens = lambda x  : ''.join(filter(lambda char: char not in special_characters, x))

            out_faq_dict = {
                'Question' : [], 'Answer' : []
            }
            for _qa_pair in qa_pairs:

                if len(_qa_pair) > 2:
                    # Multiple similar questions - one answers
                    _q = '?'.join(_qa_pair[:-1]) + '?'
                    _a = _qa_pair[-1]
                else:
                    _q, _a = _qa_pair
                    _q += '?'


                out_faq_dict['Question'].append(filter_out_tokens(_q).replace('\n',' ').strip())
                out_faq_dict['Answer'].append(filter_out_tokens(_a).replace('\n',' ').strip())


            out_path = f"{out_base_path}/{'/'.join(_faq.split('/')[-2:])}.json"
            with safe_open_w(out_path) as f_out:
                json.dump(out_faq_dict, f_out, ensure_ascii=False)
    

if __name__ == "__main__":


    parser = argparse.ArgumentParser(prog='FAQ-Parser',
                                    description='Find and parse FAQs.')

    
    parser.add_argument('--faq_path', type=str, help='Base path for searching or loading FAQ files.')
    parser.add_argument('--out_path', type=str, help='Path for storing results.')
    parser.add_argument('--num_options', type=int, default=3, help='Number of options to generate for each question.')
    parser.add_argument('--parse_html_documents', action='store_true', default=False, help="Parse FAQ pages into json.")
    parser.add_argument('--load_faqs', action='store_true', default=False, help="Load all FAQs from the specified directory.")
    parser.add_argument('--create_options', action='store_true', default=False, help="Whether to create options for each question in the dataset.")
    parser.add_argument('--group_by_lang', action='store_true', default=False, help="Divide FAQs by language (Italian/Enlish).")
    args = parser.parse_args()
    

    if args.parse_html_documents:
        parser_config = {
            'base_path' : args.faq_path,
            'out_path' : args.out_path
        }
        parse_faqs(**parser_config)

    if args.load_faqs or args.create_options:
        loader_config = {
            'faqs_directory' : args.faq_path,
            'out_path' : args.out_path,
            'create_options' : args.create_options,
            'num_options' : args.num_options
        }
        load_faqs_into_ds(**loader_config)

    if args.group_by_lang:
        lang_divider_config = {
            'faq_in_file' : args.faq_path,
            'faq_out_file' : args.out_path
        }
        divide_faq_by_lang(**lang_divider_config)

