# Master Thesis workspace repository

The intention of this repository is to provide a LaTeX workspace for
writing and revieving my master thesis. This readme.md file indicates the
connections between LaTeX files and compiled .pdf file in sequential manner.

The configurations used in this master thesis are compliant with editorial
guidelines from the year 2018 which are available on the [university
website](https://pg.edu.pl/documents/8597924/15531473/ZR%2022-2018)
(document contains guidelines in Polish and English). Some inspiration
was taken also from [guidelines from the year
2014](https://eti.pg.edu.pl/documents/1115629/0/zarz%C4%85dzenie%20wytyczne%20pracy)
which are more detailed than the latest version.

## Table of Contents

```
1. pdf/title-page.pdf
2. pdf/author-statement.pdf
3. misc/abstract-en.tex
4. misc/list-of-symbols.tex
5. chapters/00_Introduction.tex
6. chapters/01_Research_Goal.tex
7. chapters/02_Theoretical_Background.tex
8. chapters/03_Related_Work.tex
9. chapters/04_Preparations_to_Experiments.tex
10. chapters/05_Experiments.tex
11. chapters/06_Summary.tex
12. config/bibliography.bib
13. figures/
14. tables/
```

## Chapter 0 - Introduction



## Chapter 1 

The only supported language is Polish, however it shouldn't be very hard
to adjust the template for writing in English. Polish requires
additional options, so some of the changes is just removing
Polish-specific settings.

### Then why is this readme in English?

Mostly out of habit, as I've never experienced any
`addBag/dodajWorek`:tm: issues during my studies, so I always tend to
write code and documentation in English even if Polish here makes more
sense.

## Setup

Clone the repo, and make sure that `main.tex` compiles into a PDF on
your system. The template was successfully tested against TeXLive 2022.

### Overleaf

If you prefer working in Overleaf, here you can access the files for
this template: <https://www.overleaf.com/read/mngqxzfvdhpk>.

## How to use?

This repo should be treated a sample project, so you're supposed to copy
the files and edit them freely. The most of the actual settings live in
`config/preamble.sty`. To be compliant with PG, you probably don't want
to modify existing settings there unless you're fixing something (pull
request is welcomed in this case).

Depending on your needs, you might want to include more packages and
settings in this file to get access to some more niche LaTeX features
provided by packages not included in the template by default.

## Structure and conventions

The below diagram describes all important parts of the template. You
should also take a look at the notes in the example chapter which
describes how to add figures, tables, and citations along with some good
practices. It's a good idea to analyze both the raw TeX, and the
generated PDF (you can get the PDF easily from the Overleaf linked
above).

```
main.tex
 \_ Entry point of the template, start here to understand how the rest
    of the files is referenced
config
  \_ preamble.sty      # The heart of the template with most settings
  \_ bibliography.bib  # Put your bibliography positions here
  \_ macros.sty        # Define your macros here
chapters # A TeX file per chapter
  \_ 01.tex
  \_ 02.tex
figures
  \_ Graphics, images etc.
misc
  \_ TeX files which are not chapters but are part of the thesis
pdf
  \_ Here you should put PDFs which will be included as separate pages
     into your thesis
tables
  \_ Big and complex tables can be defined here and with `\input{}`
     included in chapters
```

## Misc

In some places, you can notice a bit strange formatting with LaTeX
comments `%` at the end of lines. This was done to prevent automatic
wrapping of lines by [stkb/Rewrap](https://github.com/stkb/Rewrap) VS
Code extension.

## See also

* [jachoo/pg-beamer](https://github.com/jachoo/pg-beamer)\
  A PG template for creating LaTeX presentations

* [splaw1k/PG_LaTeX_Templates](https://github.com/splaw1k/PG_LaTeX_Templates)\
  Another PG thesis template available on GitHub

* [typografia.info](https://typografia.info/podstawy)\
  Writing a thesis is not writing a book, but some typographic knowledge
  is nice to have if you decided to be a LaTeX guy (some people *really*
  prefer Word)

* [James-Yu/LaTeX-Workshop](https://github.com/James-Yu/LaTeX-Workshop)\
  If you're new to LaTeX and already familiar with VS Code, you can try
  this extension before installing full-fledged LaTeX "IDE"

## Contributing

All fixes, proposed updates, or comments are welcome!
