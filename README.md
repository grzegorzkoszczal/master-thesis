# Master Thesis workspace

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

This section is created as an overview on the relations of .tex files with
compiled .pdf file, rather than real ToC. For the actual Table of Contents,
please refer to main.pdf file and see the compiled master thesis or see
the chapters below.

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

## Abstract

Short insight on master thesis, keywords and field of study. Next we can see
table of contents of the master thesis and list of important symbols and
abbreviations.

## Chapter 0 - Introduction

Introduction chapter gives a short overview on global shares of energy usage
in ICT sector and needs of reliable power measurement methods in HPC.

## Chapter 1 - Research goal

### 1.1 Purpose and research question

Main goal of the master thesis

### 1.2 Scope and limitation

Constraint on the research scope put by the devices and measurements
tools used.

### 1.3 Project requirements

Requirements set for computation servers, benchmark applications chosen for the
tests, and overall preparation for the experiments

## Chapter 2 - Theoretical background

More theory about used measurements software (RAPL, NVML) and hardware
(Yokogawa), as well as benchmark applications.

### 2.1 Measurement software

#### 2.1.1 Intel RAPL

Short explanation about CPU-specific measurement tool, utilized by Linux Perf
used in later tests.

#### 2.1.2 NVIDIA NVML

Short explanation about GPU-specific measurement tool.

### 2.1 Measurement hardware

#### 2.2.1 Yokogawa WT310E

Short explanation about hardware measurement tool, that collect the data of
power draw of the entire node.

### 2.3 Benchmark applications

Introduction to NAS Parallel Benchmarks - who wrote them, why are they popular,
widely used and reliable, what different implementations are available in the
benchmarks suite, what benchmark problems do they compute (kernels and
pseudo-applications) and finally, what are the class sizes for each benchmark.

#### 2.3.1 NPB for CPU, C++ with OMP

Overview on C++ as a programming language that is a reliable choice for
writing a code with focus on performance. Brief explanation of OpenMP
directives for parallel programming. Advantages of the overall benchmark
in chosing the configurations during tests.

#### 2.3.2. NPB for CPU, Fortran with MPI

Brief insight on advantages of Fortran programming language in numerical
and scientific computing. Advantages of MPI in both parallel and distributed
computing (mainly inter-node communication support).
#### 2.3.3. NPB for GPU, CUDA

Short description of CUDA, a parallel computing platform and application
programming interface designed to utilize the resources of GPUs. Pros and
cons of the benchmarks with a commentary on the issues and solutions that
emerged during the usage of this benchmark suite

#### 2.3.4. Custom Deep Neural Networks model

More detailed explanation of this benchmark comes from the fact, that it has
been written for the sole purpose of being used in this master thesis, in order
to fill the need for multi-GPU, multi-node benchmark. At first are explained
the advantages of Python as a popular general-purpose programming language with
many modules that helps works in fields like web development, data science or
artificial intelligence. Next, the deep neural networks are explained, such as
TensorFlow, Keras and Horovod, that particularly focues on training models in
efficient parallel and distributed manner. Next, the model itself is explained
in more details - the problem it solves, how the dataset was aqcuired and why
it is valuable as a benchmark in this work

## Chapter 3 - Related work

This chapter describes the workflow of experiments done in the research papers,
that broach the topics similar to the goal of this master thesis. This part
has been written as an requirement for Master Thesis Seminar I
(the "state-of-the-art" chapter was required to pass that subject) and it was
graded 5.0 for the contents, so I am fairly confident, that there is little to
change in that terms.

#### 3.1. "A comparative study of methos for measurement of energy of computing"

#### 3.2. "Verified instruction-level energy consumption measurement for NVIDIA GPUs"

#### 3.3. "Measuring GPU power with the K20 built-in sensor"

## Chapter 4 - Preparations to experiments

### 4.1 [Placeholder]

## Chapter 5 - Experiments

### 5.1 [Placeholder]

## Chapter 6 - Summary and future work

### 6.1 Summary

#### 6.2 Future work

## To-do list:

I highly encourage my supervisor, Dr. Czarnul to periodicaly check the updates
on this master thesis and write back the feedback on what is needed to be
written and/or changed.

1. Field of science and technology in accordance with OECD requirements