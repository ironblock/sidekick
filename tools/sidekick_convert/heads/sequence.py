"""The checkpoint's own sequence-classification head (classifiers and
rerankers, docs/DECISIONS.md D28).

The backbone loads the checkpoint's task class (e.g.
BertForSequenceClassification), so the head is exactly the checkpoint's:
BERT's pooler (dense + tanh on [CLS]) and classifier, ModernBERT's head and
pooling, and so on. The artifact returns raw logits, (1, num_labels) in
id2label order; the activation is the manifest's problem_type, applied by
sidekick, never in the graph. A reranker is the same head with one label and
a token_type_ids port.
"""

import dataclasses


@dataclasses.dataclass
class SequenceClassification:
    output: str = "logits"
    task: str = "sequence-classification"
    num_labels: int = None
    labels: list = None

    def bind(self, backbone):
        config = backbone.config
        self.num_labels = int(config.num_labels)
        self.labels = [config.id2label[i] for i in range(self.num_labels)]
        return self

    def register(self, w, seq):
        pass

    def forward(self, w, x, backbone):
        return backbone.call(w, x).logits.reshape(1, self.num_labels)

    def reference(self, outputs):
        return outputs.logits[0].double().numpy()
