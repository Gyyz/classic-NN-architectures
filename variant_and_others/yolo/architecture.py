"""
Yolo — One-stage detector predicting boxes and classes directly.
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
    papers_data = [("You Only Look Once: Unified, Real-Time Object Detection", "Redmon et al.", 2016, "https://arxiv.org/abs/1506.02640")]
    papers = [Paper(title=t, authors=a, year=y, url=u) for t, a, y, u in papers_data]
    return ArchitectureDescription(name='yolo', title='Yolo', category='Detection', summary='One-stage detector predicting boxes and classes directly.', papers=papers)

if __name__ == '__main__':
    d = describe()
    print(d.title)
    print(d.category)
    print(d.summary)
    for p in d.papers:
        print(p.title, p.authors, p.year, p.url)
