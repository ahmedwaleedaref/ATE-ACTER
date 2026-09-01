def load_gold_list_into_set(path : str):
    gold_set : set[str] = set()
    with open(path , "r") as file:
        for line in file : 
            line_list = line.split("\t")
            gold_set.add(line_list[0])
    return gold_set

def score_list(predicted_unique : set[str] , gold_list : set[str] ):
    """
    input is a set of string , will be compared againist gold set also of string to compute recall and precision then f-score 
    """
    # gold key is never empty — an empty gold set means a bad path or a bad loader, not a bad model
    assert len(gold_list) > 0 , "gold list is empty"
    # an empty prediction is a legitimate model state (early training), not an error
    if len(predicted_unique) == 0 :
        return 0.0 , 0.0 , 0.0

    # islower() is False for strings with no cased characters ("2020", "%", "hr")
    assert all(
        isinstance(x,str) and x == x.lower() for x in predicted_unique
    ), "All elements must be lowercase strings"
    
    number_of_spans_predicted_correctly : int = 0 
    
    for predicted_span in predicted_unique :
        if predicted_span in gold_list : 
            number_of_spans_predicted_correctly += 1 
            
    Precision : float = (number_of_spans_predicted_correctly)  / (len(predicted_unique))          
    
    number_of_gold_span_predicted_correctly : int = 0 
    
    for gold_span in gold_list : 
        if gold_span in predicted_unique : 
            number_of_gold_span_predicted_correctly += 1
            
    Recall : float = (number_of_gold_span_predicted_correctly) / (len(gold_list))
    
    # zero overlap: both sets non-empty but nothing matches — the empty-set check above does not cover this
    if Precision + Recall == 0.0 :
        return Precision , Recall , 0.0

    f1_score = 2 * (Recall*Precision) / (Recall+Precision) 
    return Precision , Recall , f1_score 


# TOKENISED key only. The non-tokenised variant sits beside it with a near-identical
# name and splits hyphens and internal punctuation differently ("public prosecutor's
# office" vs "public prosecutor 's office"). decode() joins original dataset tokens
# with spaces, so it can only ever match the tokenised form. Wrong file = ~1% of the
# key silently scored as false positive + false negative, no crash. data_layout.md §4, §6.
GOLD_PATH = "/home/ahmed-walled/Projects/ATE-ACTER/ate-acter/data/raw/ACTER/en/corp/annotated/annotations/unique_annotation_lists/corp_en_tokenised_terms_nes.tsv"

print(score_list(load_gold_list_into_set(GOLD_PATH) , load_gold_list_into_set(GOLD_PATH)))