from .teds import TEDS


class TEDSMetric:
    def __init__(self, num_workers=1, structure_only=False):
        self.evaluator = TEDS(n_jobs=num_workers, structure_only=structure_only)

    def __call__(self, pred_htmls, label_htmls):
        assert len(pred_htmls) == len(label_htmls)
        pred_jsons = {idx: pred_html for idx, pred_html in enumerate(pred_htmls)}
        label_jsons = {idx: dict(html=label_html) for idx, label_html in enumerate(label_htmls)}
        scores = self.evaluator.batch_evaluate(pred_jsons, label_jsons)
        scores = [scores[idx] for idx in range(len(pred_htmls))]
        return scores
