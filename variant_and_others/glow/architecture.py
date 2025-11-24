"""
Glow — Flow-based model with actnorm and invertible 1x1 convs.
Category: Generative
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
    papers_data = [("Glow: Generative Flow with Invertible 1x1 Convolutions", "Kingma and Dhariwal", 2018, "https://arxiv.org/abs/1807.03039")]
    papers = [Paper(title=t, authors=a, year=y, url=u) for t, a, y, u in papers_data]
    return ArchitectureDescription(name='glow', title='Glow', category='Generative', summary='Flow-based model with actnorm and invertible 1x1 convs.', papers=papers)

if __name__ == '__main__':
    d = describe()
    print(d.title)
    print(d.category)
    print(d.summary)
    for p in d.papers:
        print(p.title, p.authors, p.year, p.url)
