def decode(tokens: list[str], labels: list[str], scheme: str) -> list[tuple[int, int]]:
    """
    Decode an IOB label sequence for ONE sentence into term spans.

    tokens, labels : parallel lists, one entry per dataset token
    scheme         : annotation scheme; only "bio" is implemented
    
    Dangling "I" (sentence-initial, or following "O") opens no span and is
    dropped. This matches seqeval's mode='strict', scheme=IOB2, which the
    span scorer is cross-checked against. Gold data is IOB2 — measured: 0
    I-after-O in 222,281 tokens, 3 sentence-initial I, all terms split by a
    spurious sentence boundary (wind_en_01 x2, htfl_en_171 x1). Model output
    is unconstrained, so the policy is load-bearing there.

    Returns (start, end) pairs with end EXCLUSIVE: tokens[start:end] is the term.
    Spans never cross a sentence boundary, because one call sees one sentence.
    """
    assert scheme == "bio", f"unsupported scheme: {scheme}"
    assert len(tokens) == len(labels), f"{len(tokens)} tokens vs {len(labels)} labels"

    span_list: list[tuple[int, int]] = []
    curr_pair: list[int] = []

    for index, label in enumerate(labels):

        if label == "O":
            if len(curr_pair) != 0:
                span_list.append((curr_pair[0], index))
                curr_pair = []

        elif label == "B":
            if len(curr_pair) != 0:
                span_list.append((curr_pair[0], index))
                curr_pair = []

            if index == len(labels) - 1:
                span_list.append((index, index + 1))
                curr_pair = []
            else:
                curr_pair.append(index)

        elif label == "I":
            if len(curr_pair) != 0 and index == len(labels) - 1:
                span_list.append((curr_pair[0], index + 1))
                curr_pair = []

        else:
            raise ValueError(f"unknown label {label!r} at index {index}")

    return span_list

def count_invalid_tags(labels: list[str], scheme: str) -> dict[str, int]:
    """
    Count the IOB2 violations in ONE sentence's label sequence.

    labels : one label per DATASET token -- the recovered token labels from model predictions .
    scheme : only "bio" is implemented. NOBI has different rules and must not
             reuse this count.

    IOB2 has exactly two illegal patterns, both about the PREDECESSOR:
        - "I" with no predecessor       (sentence-initial)
        - "I" whose predecessor is "O"
    "I" after "B" and "I" after "I" are both legal.

    Gold is strict IOB2 by measurement -- data_layout.md section 3. Summed over a
    domain: corp (0, 0), equi (0, 0), wind (2, 0), htfl (1, 0), with 0 I-after-O
    across all 222,281 tokens and three sentence-initial "I" corpus-wide. So any
    nonzero count from a model is entirely the model's, with no contribution from
    the data.
    """
    assert scheme == "bio", f"unsupported scheme: {scheme}"

    i_sentence_initial: int = 0
    i_after_o: int = 0

    for index, label in enumerate(labels):

        if label not in ("O", "B", "I"):
            raise ValueError(f"unknown label {label!r} at index {index}")

        if label != "I":
            continue

        if index == 0:
            i_sentence_initial += 1
        elif labels[index - 1] == "O":
            i_after_o += 1

    return {
        "i_sentence_initial": i_sentence_initial,
        "i_after_o": i_after_o,
    }

def encode(tokens: list[str], spans: list[tuple[int, int]], scheme: str ) : 
    """
    encode is inverse of decode it takes tokens and spans and generate the labels . 
    this is handling only for IBO case NBIO for weak 3 . 
    """
    assert scheme == "bio", f"unsupported scheme: {scheme}"
    assert all(0 <= s < e <= len(tokens) for s, e in spans)
    
    labels : list[str] = ["O"] * len(tokens)
    
    
    for span in spans : 
        first_index = span[0]
        last_index  = span[1]
        #you should loop from (first_index to last_index[
        for i in range(first_index , last_index) : 
            if i == first_index :
                labels[i] = "B"
            else : 
                labels[i] = "I"
    return labels