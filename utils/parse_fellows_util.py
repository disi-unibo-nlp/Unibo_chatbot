import json
import os
from tqdm import tqdm



def parse_fellows():   

    data_path = 'data/unibo_web/unibo.sitemap.xml'
    line_tok = '<url>'
    start_name = '<loc>'
    end_name = '</loc>'

    content = []
    with open(data_path, 'r') as f_in:
        content = f_in.read()


    fellows = []
    parser_data = content.split(line_tok)
    parsed_authors = [auth.split(start_name)[-1].split(end_name)[0] for auth in  parser_data]

    fellow_sites = [auth for auth in parsed_authors if 'sitoweb' in auth]
    fellow_dict = {'sites' : fellow_sites}

    out_file = 'data/unibo_web/fellow_sites.json'
    with open(out_file, 'w') as f_out:
        json.dump(fellow_dict, f_out)

    output_tree_path = 'data/unibo_web/fellows_tree/'

    for auth_site in tqdm(fellow_sites):
        _ = os.system(f'wget -w 0 --no-parent {auth_site} -P {output_tree_path}')

    print("DONE")


if __name__ == "__main__":
    parse_fellows()