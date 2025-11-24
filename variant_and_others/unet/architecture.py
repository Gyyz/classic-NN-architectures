"""
Unet — Encoder-decoder with skip connections for segmentation.
Category: Segmentation
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
    papers_data = [("U-Net: Convolutional Networks for Biomedical Image Segmentation", "Ronneberger et al.", 2015, "https://arxiv.org/abs/1505.04597")]
    papers = [Paper(title=t, authors=a, year=y, url=u) for t, a, y, u in papers_data]
    return ArchitectureDescription(name='unet', title='Unet', category='Segmentation', summary='Encoder-decoder with skip connections for segmentation.', papers=papers)

if __name__ == '__main__':
    d = describe()
    print(d.title)
    print(d.category)
    print(d.summary)
    for p in d.papers:
        print(p.title, p.authors, p.year, p.url)
