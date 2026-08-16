"""Fetch and cache Gene Ontology molecular-function annotations from UniProt."""

import json
import os

import requests

UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"


def fetch_go_molecular_function(
    accession: str, session: requests.Session | None = None, timeout: int = 10
) -> list[str]:
    """Fetch GO molecular-function term names for a single UniProt accession.

    Returns an empty list if the accession has no entry or no molecular-function
    GO cross-references -- proteins with sparse annotations are a normal part of
    this dataset, not an error case.
    """
    session = session or requests.Session()
    response = session.get(UNIPROT_ENTRY_URL.format(accession=accession), timeout=timeout)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    entry = response.json()

    terms = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        for prop in xref.get("properties", []):
            if prop.get("key") == "GoTerm" and prop.get("value", "").startswith("F:"):
                terms.append(prop["value"][2:])
    return terms


class GOAnnotationCache:
    """On-disk JSON cache mapping a UniProt accession to its GO-MF term names."""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache: dict[str, list[str]] = json.load(f)
        else:
            self._cache = {}

    def get_or_fetch(
        self, accession: str, session: requests.Session | None = None
    ) -> list[str]:
        if accession not in self._cache:
            self._cache[accession] = fetch_go_molecular_function(accession, session=session)
            self._save()
        return self._cache[accession]

    def _save(self) -> None:
        directory = os.path.dirname(self.cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)
