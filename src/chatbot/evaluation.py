def calculate_mrr(responses, ground_truth):
    mrr = []
    # iterate through a list of ground truth, 
    # each ground truth will have a list of retriever
    for response_list, truth in zip(responses, ground_truth):
        if truth in response_list:
            rank = response_list.index(truth) + 1
            mrr.append(1 / rank)
        else:
            mrr.append(0)
    return sum(mrr) / len(mrr)


def hit_rate_at_k(predictions, ground_truth, k):
    hits = 0
    for preds, truth in zip(predictions, ground_truth):
        if truth in preds[:k]:
            hits += 1
    return hits / len(ground_truth)