"""Localization metrics shared by the learned ranker and baselines."""


def localization_metrics(rankings: list[list[int]], culprits: list[int]) -> dict[str, float]:
    if not rankings or len(rankings) != len(culprits):
        raise ValueError("provide one ranking per culprit and at least one run")
    reciprocal, top1, top3 = [], 0, 0
    for ranking, culprit in zip(rankings, culprits):
        if len(set(ranking)) != len(ranking):
            raise ValueError("ranking contains duplicate steps")
        rank = ranking.index(culprit) + 1 if culprit in ranking else None
        top1 += rank == 1
        top3 += rank is not None and rank <= 3
        reciprocal.append(1 / rank if rank else 0)
    count = len(rankings)
    return {"recall@1": top1 / count, "recall@3": top3 / count, "mrr": sum(reciprocal) / count}
