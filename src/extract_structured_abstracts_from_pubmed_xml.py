"""
Extract structured abstracts from PubMed articles from FTP.

This script parses the XML files from the FTP server and extracts structured
abstract sections when AbstractText nodes carry NlmCategory or Label attributes.

If you want to download the XML files from the FTP server, you can use the following command:
wget ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/*.xml.gz
wget ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/*.xml.gz

Please check https://pubmed.ncbi.nlm.nih.gov/download/ for more details of the XML files from PubMed FTP server.
"""

import argparse
import json
import os

import lxml.etree as ET
import pandas as pd
from tqdm import tqdm

from utils import *


def _abstract_text_content(abst_el):
    return "".join(abst_el.itertext()).strip()


def _section_key_for_abstract_text(abst_el):
    nlm = abst_el.get("NlmCategory")
    if nlm and str(nlm).strip():
        return str(nlm).strip()
    label = abst_el.get("Label")
    if label and str(label).strip():
        return str(label).strip()
    return None


@timing_decorator
def extract_structured_abstracts(path_xml, path_out, logger=None):
    """Parse one PubMed XML file and write a TSV of structured abstract fields per article."""
    print_or_log(logger, "* extracting structured abstracts from %s" % path_xml)

    parser = ET.XMLParser(recover=True)
    tree = ET.parse(path_xml, parser=parser)
    root = tree.getroot()

    pubmed_articles = root.findall(".//PubmedArticle")
    print_or_log(logger, "* found %s pubmed_articles" % (len(pubmed_articles)))
    pubmed_data = []
    for pubmed_article in tqdm(pubmed_articles):
        pmid = pubmed_article.findtext(".//PMID")
        abst_elements = pubmed_article.findall(".//AbstractText")

        has_structured = False
        for abst in abst_elements:
            nlm = abst.get("NlmCategory")
            lab = abst.get("Label")
            if (nlm and str(nlm).strip()) or (lab and str(lab).strip()):
                has_structured = True
                break

        if not has_structured:
            pubmed_entry = {
                "pmid": pmid,
                "contain_structured": False,
                "structured_key": "",
                "structured_abstract": json.dumps({}),
            }
            pubmed_data.append(pubmed_entry)
            continue

        key_order = []
        sections = {}
        for abst in abst_elements:
            key = _section_key_for_abstract_text(abst)
            if key is None:
                continue
            text = _abstract_text_content(abst)
            if key not in sections:
                key_order.append(key)
                sections[key] = text
            else:
                prev = sections[key]
                sections[key] = (prev + " " + text).strip() if prev else text

        structured_key = "|".join(key_order)
        pubmed_entry = {
            "pmid": pmid,
            "contain_structured": True,
            "structured_key": structured_key,
            "structured_abstract": json.dumps(sections, ensure_ascii=False),
        }
        pubmed_data.append(pubmed_entry)

    df = pd.DataFrame(pubmed_data)
    save_df(df, path_out, logger=logger)
    return df


@timing_decorator
def extract_structured_abstracts_from_folder(
    xml_gz_file_folder, folder_out, end_pattern=".xml.gz", logger=None
):
    xml_gz_file_paths = get_file_paths(
        xml_gz_file_folder, end_pattern=end_pattern, logger=logger
    )
    for xml_gz_file_path in xml_gz_file_paths:
        file_name = os.path.basename(xml_gz_file_path).replace(end_pattern, ".tsv")
        path_out = os.path.join(folder_out, file_name)
        extract_structured_abstracts(xml_gz_file_path, path_out, logger=logger)
    print_or_log(
        logger,
        f"* extracted structured abstracts from {len(xml_gz_file_paths)} {end_pattern} files!",
    )
    return xml_gz_file_paths


@timing_decorator
def main():
    parser = argparse.ArgumentParser(
        description="Extract structured abstracts for PubMed articles from FTP XML."
    )
    parser.add_argument(
        "--log_file_path", "-log", type=str, required=True, help="log file path"
    )
    parser.add_argument(
        "--xml_gz_file_folder", "-gz", type=str, required=True, help="input file folder"
    )
    parser.add_argument(
        "--folder_out", "-fo", type=str, required=True, help="output file folder"
    )

    args = parser.parse_args()
    log_file_path = args.log_file_path
    xml_gz_file_folder = args.xml_gz_file_folder
    folder_out = args.folder_out

    logger = setup_logger(log_file_path)

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    extract_structured_abstracts_from_folder(
        xml_gz_file_folder, folder_out, end_pattern=".xml.gz", logger=logger
    )


if __name__ == "__main__":
    main()
