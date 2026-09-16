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
    if scheme == "nobi": #Talaat may change the way if works 
        from src.eval.nobi import decode_nobi

        return decode_nobi(tokens, labels)
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