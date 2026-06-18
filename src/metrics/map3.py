def average_prediction_score3(prediction:list, ground_truth:str) -> float:
    """
    Average Precision Score@3
   
    Example:
        If the correct answer is: A
        Then the following predictions would receive:
        A B C   → highest score
        B A C   → Relatively lower score
        C D A   → lowest score
    """
    
    if not prediction:
        return 0.0

    score = 0.0
    for i, pred in enumerate(prediction[:3]):
        if pred == ground_truth:
            score += 1 / (i + 1)  

    return score

def mean_average_precision3(predictions:list, ground_truths:list) -> float:
    """
    Mean Average Precision@3 (MAP@3) Score
    """
    
    if not predictions or not ground_truths:
        return 0.0

    total_score = 0.0
    for pred, gt in zip(predictions, ground_truths):
        total_score += average_prediction_score3(pred, gt)

    return total_score / len(predictions)