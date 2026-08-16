import json

from warl.go_annotations import GOAnnotationCache, fetch_go_molecular_function


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return self._response


UNIPROT_PAYLOAD = {
    "uniProtKBCrossReferences": [
        {
            "database": "GO",
            "id": "GO:0005515",
            "properties": [
                {"key": "GoTerm", "value": "F:protein binding"},
                {"key": "GoEvidenceType", "value": "IEA:InterPro"},
            ],
        },
        {
            "database": "GO",
            "id": "GO:0008270",
            "properties": [{"key": "GoTerm", "value": "F:zinc ion binding"}],
        },
        {
            "database": "GO",
            "id": "GO:0006351",
            "properties": [{"key": "GoTerm", "value": "P:transcription, DNA-templated"}],
        },
        {
            "database": "InterPro",
            "id": "IPR000123",
            "properties": [{"key": "EntryName", "value": "Some_domain"}],
        },
    ]
}


def test_fetch_go_molecular_function_extracts_only_mf_terms():
    session = _FakeSession(_FakeResponse(200, UNIPROT_PAYLOAD))

    terms = fetch_go_molecular_function("P12345", session=session)

    assert terms == ["protein binding", "zinc ion binding"]
    assert session.calls == ["https://rest.uniprot.org/uniprotkb/P12345.json"]


def test_fetch_go_molecular_function_returns_empty_on_404():
    session = _FakeSession(_FakeResponse(404))

    terms = fetch_go_molecular_function("NOTAREALID", session=session)

    assert terms == []


def test_cache_fetches_once_and_persists_to_disk(tmp_path):
    cache_path = tmp_path / "go_cache.json"
    session = _FakeSession(_FakeResponse(200, UNIPROT_PAYLOAD))
    cache = GOAnnotationCache(str(cache_path))

    first = cache.get_or_fetch("P12345", session=session)
    second = cache.get_or_fetch("P12345", session=session)

    assert first == ["protein binding", "zinc ion binding"]
    assert second == first
    assert len(session.calls) == 1  # second call served from the in-memory cache

    with open(cache_path) as f:
        on_disk = json.load(f)
    assert on_disk == {"P12345": ["protein binding", "zinc ion binding"]}


def test_cache_loads_existing_file_on_init(tmp_path):
    cache_path = tmp_path / "go_cache.json"
    cache_path.write_text(json.dumps({"P99999": ["kinase activity"]}))
    cache = GOAnnotationCache(str(cache_path))

    terms = cache.get_or_fetch("P99999", session=_FakeSession(_FakeResponse(500)))

    assert terms == ["kinase activity"]  # served from disk, never touched the (failing) session
