"""
Gpt2 — Scaled GPT with WebText training.
Category: Transformer
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Paper:
    title: str
    authors: str
    year: int
    url: str

@dataclass
class ArchitectureDescription:
    name: str
    title: str
    category: str
    summary: str
    papers: List[Paper]

def describe() -> ArchitectureDescription:
    papers_data = [("Language Models are Unsupervised Multitask Learners", "Radford et al.", 2019, "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf")]
    papers = [Paper(title=t, authors=a, year=y, url=u) for t, a, y, u in papers_data]
    return ArchitectureDescription(name='gpt2', title='Gpt2', category='Transformer', summary='Scaled GPT with WebText training.', papers=papers)

if __name__ == '__main__':
    d = describe()
    print(d.title)
    print(d.category)
    print(d.summary)
    for p in d.papers:
        print(p.title, p.authors, p.year, p.url)
