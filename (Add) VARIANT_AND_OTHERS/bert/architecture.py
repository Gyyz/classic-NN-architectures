"""
Bert — Bidirectional transformer pretraining with masked language modeling.
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
    papers_data = [("BERT: Pre-training of Deep Bidirectional Transformers", "Devlin et al.", 2018, "https://arxiv.org/abs/1810.04805")]
    papers = [Paper(title=t, authors=a, year=y, url=u) for t, a, y, u in papers_data]
    return ArchitectureDescription(name='bert', title='Bert', category='Transformer', summary='Bidirectional transformer pretraining with masked language modeling.', papers=papers)

if __name__ == '__main__':
    d = describe()
    print(d.title)
    print(d.category)
    print(d.summary)
    for p in d.papers:
        print(p.title, p.authors, p.year, p.url)
