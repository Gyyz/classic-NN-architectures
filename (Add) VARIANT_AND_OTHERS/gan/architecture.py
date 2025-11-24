"""
Gan — Generative Adversarial Network with generator and discriminator.
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
    papers_data = [("Generative Adversarial Nets", "Goodfellow et al.", 2014, "https://papers.nips.cc/paper_files/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html")]
    papers = [Paper(title=t, authors=a, year=y, url=u) for t, a, y, u in papers_data]
    return ArchitectureDescription(name='gan', title='Gan', category='Generative', summary='Generative Adversarial Network with generator and discriminator.', papers=papers)

if __name__ == '__main__':
    d = describe()
    print(d.title)
    print(d.category)
    print(d.summary)
    for p in d.papers:
        print(p.title, p.authors, p.year, p.url)
