"""
Gpt — Autoregressive transformer for next-token prediction.
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
    papers_data = [("Improving Language Understanding by Generative Pre-Training", "Radford et al.", 2018, "https://openai.com/research/language-unsupervised")]
    papers = [Paper(title=t, authors=a, year=y, url=u) for t, a, y, u in papers_data]
    return ArchitectureDescription(name='gpt', title='Gpt', category='Transformer', summary='Autoregressive transformer for next-token prediction.', papers=papers)

if __name__ == '__main__':
    d = describe()
    print(d.title)
    print(d.category)
    print(d.summary)
    for p in d.papers:
        print(p.title, p.authors, p.year, p.url)
