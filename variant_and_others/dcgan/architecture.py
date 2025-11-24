"""
Dcgan — Convolutional GAN architecture for images.
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
    papers_data = [("Unsupervised Representation Learning with Deep Convolutional GANs", "Radford et al.", 2015, "https://arxiv.org/abs/1511.06434")]
    papers = [Paper(title=t, authors=a, year=y, url=u) for t, a, y, u in papers_data]
    return ArchitectureDescription(name='dcgan', title='Dcgan', category='Generative', summary='Convolutional GAN architecture for images.', papers=papers)

if __name__ == '__main__':
    d = describe()
    print(d.title)
    print(d.category)
    print(d.summary)
    for p in d.papers:
        print(p.title, p.authors, p.year, p.url)
