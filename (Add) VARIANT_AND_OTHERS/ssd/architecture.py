"""
Ssd — Single Shot detector with multi-scale feature maps.
Category: Detection
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
    papers_data = [("SSD: Single Shot MultiBox Detector", "Liu et al.", 2016, "https://arxiv.org/abs/1512.02325")]
    papers = [Paper(title=t, authors=a, year=y, url=u) for t, a, y, u in papers_data]
    return ArchitectureDescription(name='ssd', title='Ssd', category='Detection', summary='Single Shot detector with multi-scale feature maps.', papers=papers)

if __name__ == '__main__':
    d = describe()
    print(d.title)
    print(d.category)
    print(d.summary)
    for p in d.papers:
        print(p.title, p.authors, p.year, p.url)
