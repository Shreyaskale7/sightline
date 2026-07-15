"""Evaluation harness — the heart of the project.

Built before any visual retrieval. It is what lets every later change be PROVEN to help with a
number on the benchmark, instead of relying on "it feels better." Retrieval quality is treated
as a measured quantity, not an assumption.

Metrics we care about:
  Retrieval:  Recall@k, nDCG@10, MRR   (did we find the right page?)
  Generation: faithfulness, answer_correctness, citation_accuracy, abstention
              (is the answer backed by a cited page? does it correctly say "I don't know"?)
"""
