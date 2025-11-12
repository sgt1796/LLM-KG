import json, time, difflib
from pathlib import Path
from typing import Dict, Any, List

class LabelStore:
    """
    Persistent, bounded, fuzzy-deduped relation label set.

    File format (JSON):
    {
      "labels": { "produces": {"count": 42, "first": 1712345678, "last": 1712349999}, ... },
      "max_labels": 256,
      "min_promote": 2,     # require at least N observations to persist new labels
      "sim_threshold": 0.86 # fuzzy dup threshold for collapsing near-duplicates
    }
    """
    def __init__(self, path: str|Path, max_labels: int = 256, min_promote: int = 2, sim_threshold: float = 0.86):
        self.path = Path(path)
        self.max_labels = max_labels
        self.min_promote = min_promote
        self.sim_threshold = sim_threshold
        self.labels: Dict[str, Dict[str, float|int]] = {}
        self._loaded = False
        self.load()

    # -------- persistence --------
    def load(self):
        if self._loaded: return
        if self.path.exists():
            try:
                obj = json.loads(self.path.read_text(encoding="utf-8"))
                self.labels = obj.get("labels", {})
                self.max_labels = int(obj.get("max_labels", self.max_labels))
                self.min_promote = int(obj.get("min_promote", self.min_promote))
                self.sim_threshold = float(obj.get("sim_threshold", self.sim_threshold))
            except Exception:
                # fall back to empty if corrupted
                self.labels = {}
        self._loaded = True

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "labels": self.labels,
            "max_labels": self.max_labels,
            "min_promote": self.min_promote,
            "sim_threshold": self.sim_threshold,
        }
        self.path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------- operations --------
    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(s.casefold().strip().split())

    def _closest(self, s: str) -> tuple[str|None, float]:
        """Return (existing_label, similarity) best match, or (None, 0.0)."""
        if not self.labels: return (None, 0.0)
        s_norm = self._norm(s)
        best, score = None, 0.0
        for lab in self.labels.keys():
            r = difflib.SequenceMatcher(a=self._norm(lab), b=s_norm).ratio()
            if r > score:
                best, score = lab, r
        return best, score

    def observe(self, raw_label: str) -> str:
        """
        Observe a candidate label. Returns the canonical label we’ll count it under.
        If it's near-duplicate, fold into the closest; else track as 'pending' until promotion.
        If not labels yet, adds the new label directly.
        """
        if not raw_label: return ""
        self.load()
        now = int(time.time())
        if not self.labels:
            # First label ever: add directly
            self.labels[raw_label] = {"count": 1, "first": now, "last": now}
            return raw_label
        # If close to an existing label, fold into it immediately
        near, sim = self._closest(raw_label)
        if near and sim >= self.sim_threshold:
            meta = self.labels.setdefault(near, {"count": 0, "first": now, "last": now})
            meta["count"] = int(meta.get("count", 0)) + 1
            meta["last"] = now
            return near

        # New spelling: reserve a slot if capacity allows; promote only after min_promote
        # We store it directly but only keep it long-term once count >= min_promote.
        # If we overflow capacity, drop the rarest labels.
        meta = self.labels.setdefault(raw_label, {"count": 0, "first": now, "last": now})
        meta["count"] = int(meta.get("count", 0)) + 1
        meta["last"] = now

        # Capacity control: prune by count (ascending), then by recency.
        if len(self.labels) > self.max_labels:
            items = sorted(self.labels.items(), key=lambda kv: (int(kv[1].get("count", 0)), int(kv[1].get("last", 0))))
            # drop as many as needed to fit
            to_drop = len(self.labels) - self.max_labels
            for i in range(to_drop):
                k,_ = items[i]
                if k != raw_label:  # avoid removing the one we just observed if possible
                    self.labels.pop(k, None)

        return raw_label

    def top(self, k: int | None = None) -> List[str]:
        """Return current labels, sorted by frequency then recency. if k is None, return all."""
        self.load()
        items = sorted(self.labels.items(), key=lambda kv: (-int(kv[1].get("count", 0)), -int(kv[1].get("last", 0))))
        labs = [k for k,_ in items]
        return labs if k is None else labs[:k]
    
    def __str__(self):
        return '|'.join(self.top())

    def should_emit(self, label: str) -> bool:
        """Only emit labels with at least min_promote observations (reduce blow-up)."""
        m = self.labels.get(label)
        return bool(m) and int(m.get("count", 0)) >= self.min_promote
