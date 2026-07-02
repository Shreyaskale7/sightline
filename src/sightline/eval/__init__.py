"""Evaluation harness — the heart of the project.

Build this in Milestone 1, before any visual retrieval. It is what lets you PROVE that
every later change helped, instead of saying "it feels better." In interviews this is the
part that signals you can do AI engineering rather than just call an API.

Metrics we care about:
  Retrieval:  Recall@k, nDCG@10, MRR   (did we find the right page?)
  Generation: faithfulness, answer_correctness, citation_accuracy, abstention
              (is the answer backed by a cited page? does it correctly say "I don't know"?)
"""
