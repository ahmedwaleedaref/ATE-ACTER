from src.eval.spans import decode
from src.stats.loading import Document, load_domain

def spans_to_unique_list(sentences: list[tuple[list[str], list[tuple[int, int]]]]):
    """
    Collapse decoded spans into a contract-format unique term list.

    sentences : one (tokens, spans) pair per sentence; spans are (start, end)
                with end EXCLUSIVE, as returned by decode()

    Surface form is the dataset tokens joined by a single space, on every token
    boundary. This reproduces ACTER's tokenisation rather than repairing it, so
    it matches the TOKENISED gold key only:
        <domain>_en_tokenised_terms_<key>.tsv
    The non-tokenised variant writes "public prosecutor's office" where this
    produces "public prosecutor 's office" — scoring against it costs a false
    positive and a false negative on the same term, with no error raised.

    Lowercasing happens BEFORE deduplication, so "Bent" and "bent" collapse to
    one entry as they do in the gold key. The reverse order leaves both in the
    set and one is a guaranteed false positive.

    Returns (unique_list, n_spans, n_unique). A set cannot report how many
    occurrences collapsed into it, so the counts are returned explicitly: the
    span/type ratio is what explains the gap between exact-span F1
    (occurrence-weighted) and unique-list F1 (type-weighted).
    """
    unique_list: set[str] = set()
    n_spans: int = 0

    for tokens, spans in sentences:
        for first_index, last_index in spans:
            term = " ".join(tokens[first_index:last_index])
            unique_list.add(term.lower())
            n_spans += 1

    return unique_list, n_spans, len(unique_list)

def generate_unique_list(domain: str):
    """
    """

    sentences : list[tuple[list[str], list[tuple[int, int]]]] = []

    Domains : list[str] = [domain]

    for domain in Domains :
        Documents : list[Document] = load_domain(domain)
        for doc in Documents :
            #each doc has list of sentences
            for sentence in doc.sentences :
                token = sentence[0]
                label = sentence[1]
                span = decode(token,label,"bio")
                element = (token , span)
                sentences.append(element)
    unique_list , n_spans, uniq_lenght = spans_to_unique_list(sentences)
    return unique_list, n_spans, uniq_lenght



