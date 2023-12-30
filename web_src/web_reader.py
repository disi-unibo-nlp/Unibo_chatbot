from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.readers.base import BasePydanticReader
from llama_index.schema import Document

import logging
import mimetypes
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Type

import html2text
import langid

from tqdm import tqdm
import multiprocessing as mp

from llama_index.readers.base import BaseReader
from llama_index.readers.file.docs_reader import DocxReader, HWPReader, PDFReader
from llama_index.readers.file.epub_reader import EpubReader
from llama_index.readers.file.image_reader import ImageReader
from llama_index.readers.file.ipynb_reader import IPYNBReader
from llama_index.readers.file.markdown_reader import MarkdownReader
from llama_index.readers.file.mbox_reader import MboxReader
from llama_index.readers.file.slides_reader import PptxReader
from llama_index.readers.file.tabular_reader import PandasCSVReader
from llama_index.readers.file.video_audio_reader import VideoAudioReader
from llama_index.schema import Document


def default_file_metadata_func(file_path: str) -> Dict:
    """Get some handy metadate from filesystem.

    Args:
        file_path: str: file path in str
    """

    web_path = file_path.split('/unibo_web/')[1]
    url = f"https://{web_path}"

    return {
        "file_path": file_path,
        "url" : url,
        "file_name": os.path.basename(file_path),
        "file_type": mimetypes.guess_type(file_path)[0],
        "file_size": os.path.getsize(file_path),
        "creation_date": datetime.fromtimestamp(
            Path(file_path).stat().st_ctime
        ).strftime("%Y-%m-%d"),
        "last_modified_date": datetime.fromtimestamp(
            Path(file_path).stat().st_mtime
        ).strftime("%Y-%m-%d"),
        "last_accessed_date": datetime.fromtimestamp(
            Path(file_path).stat().st_atime
        ).strftime("%Y-%m-%d"),
    }


def is_english(sentence : str):
    lang, score = langid.classify(sentence)
    return lang == 'en'

def parse_html_data(input_file_name):
    html_parser = html2text.HTML2Text()
    html_parser.ignore_links = True

    metadata = default_file_metadata_func(str(input_file_name))

    file_data_content = open(input_file_name, 'r', errors='ignore').read()
    if is_english(file_data_content): return ""
    
    try:
        _data = html_parser.handle(file_data_content)
    except Exception as e:
        return ""

    # Remove skip characters
    parsed_text = re.sub(r'\s+', ' ', _data)

    # Handle spurious header
    if '![Foto' in parsed_text:
        # remove link to profile pic - inside the characters bla bla ![Foto <nome_docente>](<link-foto>) 
        pattern = re.compile(r'.*?\!\[Foto[^\]]+\]\([^)]+\)')
        parsed_text = re.sub(pattern, '', parsed_text).strip()
    if '[Foto' in parsed_text:
        # variant without !
        pattern = re.compile(r'.*?\[Foto[^\]]+\]\([^)]+\)')
        parsed_text = re.sub(pattern, '', parsed_text).strip()

    # Handle spurious footer
    stop_words = ['CONTATTI', 'CONTACTS','Follow Unibo', 'Segui Unibo', '### Area Riservata']
    for sw in stop_words:
        parsed_text = parsed_text.split(sw)[0]

    doc = Document(text=parsed_text, metadata=metadata or {})

    return doc

logger = logging.getLogger(__name__)




#class DirectoryWebPageReader(BasePydanticReader):
class DirectoryWebPageReader():
    """
    Simple web page reader that reads pages from directory.

    Args:
        html_to_text (bool): Whether to convert HTML to text.
            Requires `html2text` package.
        metadata_fn (Optional[Callable[[str], Dict]]): A function that takes in
            a URL and returns a dictionary of metadata.
            Default is None.
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        urls (List[str]): List of URLs to scrape.
    """

    def __init__(self,
                html_to_text: bool = False,
                recursive: bool = False,
                input_dir: Optional[str] = None,
                urls: Optional[List] = None,
                exclude: Optional[List] = None,
                metadata_fn: Optional[Callable[[str], Dict]] = None) -> None:
        

        #super(DirectoryWebPageReader, self).__init__(html_to_text=html_to_text)
        super(DirectoryWebPageReader, self).__init__()

        self.html_to_text = html_to_text
        self._metadata_fn = metadata_fn
        self._recursive = recursive
        self.file_metadata = default_file_metadata_func
        self._exclude = exclude



        if urls:
            self.urls = []
            for path in urls:
                if not os.path.isfile(path):
                    raise ValueError(f"File {path} does not exist.")
                input_file = Path(path)
                self.urls.append(input_file)
        elif input_dir:
            if not os.path.isdir(input_dir):
                raise ValueError(f"Directory {input_dir} does not exist.")
            input_dir_path = Path(input_dir)
        
            print("[LOG] Loading input files ...")
            self.urls = self._add_files(input_dir_path)

        self.urls = self.urls

        
    @classmethod
    def class_name(cls) -> str:
        return "DirectoryWebPageReader"
    
    def _add_files(self, 
                   input_dir: Path) -> List[Path]:
        """Add files."""
        all_files = set()
        rejected_files = set()

        if self._exclude is not None:
            for excluded_pattern in self._exclude:
                if self._recursive:
                    # Recursive glob
                    for file in input_dir.rglob(excluded_pattern):
                        rejected_files.add(Path(file))
                else:
                    # Non-recursive glob
                    for file in input_dir.glob(excluded_pattern):
                        rejected_files.add(Path(file))

        #file_refs: Generator[Path, None, None]
        if self._recursive:
            file_refs = Path(input_dir).rglob("*")
        else:
            file_refs = Path(input_dir).glob("*")

        for ref in file_refs:
            is_dir = ref.is_dir()
            skip_because_excluded = ref in rejected_files

            if (
                is_dir
                or skip_because_excluded
            ):
                continue
            else:
                all_files.add(ref)

        new_urls = sorted(all_files)

        if len(new_urls) == 0:
            raise ValueError(f"No files found in {input_dir}.")

        # print total number of files added
        logger.debug(
            f"> [DirectoryWebPageReader] Total files added: {len(new_urls)}"
        )

        return new_urls
    
    def __post_process_text(self, text):
        """
        Remove spaces and meaningless chaarcters.
        """

        parsed_text = re.sub(r'\s+', ' ', text)

        stop_words = ['CONTATTI', 'CONTACTS','Follow Unibo', 'Segui Unibo']
        for sw in stop_words:
            parsed_text = parsed_text.split(sw)[0]

        return parsed_text
        

    def load_data(self, 
                  load_sequentially : bool = False,
                  show_progress: bool = False,
                  ignore_links: bool = True) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
            ignore_links (bool): Whether to ignore converting links from HTML

        Returns:
            List[Document]: List of documents.
        """

        if self.html_to_text:
            html_parser = html2text.HTML2Text()
            html_parser.ignore_links = ignore_links
        
        documents = []

        files_to_process = self.urls

        if load_sequentially:
            if show_progress:
                files_to_process = tqdm(self.urls, desc="Parsing files", unit="file")

            not_converted_data = []

            for url_path in files_to_process:
                metadata: Optional[dict] = None
                if self.file_metadata is not None:
                    metadata = self.file_metadata(str(url_path))

                try:
                    # Load data from path
                    with open(url_path,'r', errors='ignore') as f_in:
                        _data = f_in.read()

                    if is_english(_data): continue

                    if self.html_to_text:
                        # Convert HTML to text
                        _data = html_parser.handle(_data)
                except ImportError as e:
                    # ensure that ImportError is raised so user knows
                    # about missing dependencies
                    raise ImportError(str(e))
                except Exception as e:
                    # otherwise, just skip the file and report the error (when file is not textual)
                    not_converted_data += [url_path]
                    continue

                cleaned_data = self.__post_process_text(_data)
                doc = Document(text=cleaned_data, metadata=metadata or {})

            documents.append(doc)

            print(f"[LOG] Not converted files: {len(not_converted_data)}/{len(files_to_process)}")
        else:
            # Process in parallel with thread pooling
            NUM_PROC = 8
            with mp.Pool(NUM_PROC) as p:
                documents = list(tqdm(p.imap(parse_html_data, files_to_process), total=len(files_to_process)))

            documents = [doc for doc in documents if doc != ""]

        for doc in documents:
            # Keep only metadata['file_path', 'url'] in both embedding and llm content
            # str, which contain extreme important context that about the chunks.
            doc.excluded_embed_metadata_keys.extend(
                [
                    "file_name",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                ]
            )
            doc.excluded_llm_metadata_keys.extend(
                [
                    "file_name",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                ]
            )

        return documents