# Arson Identifier

## Background
<p>
  <img src="https://www.gov.il/BlobFolder/office/israel_police/he/israel_police.png" alt="drawing" width="150"/>
&emsp;&emsp;&emsp;
<img src="https://upload.wikimedia.org/wikipedia/en/8/84/Bar_Ilan_seal.svg" alt="drawing" width="150"/>
</p>

This project is done in cooperation with the DIFS - Arsons laboratory.\
We aim to classify the spectra measured by DIFS - Arsons lab using GC-MS 

To date, the forensic specialist measures....

INSERT PIC herejupyter 

by using a team of experts it determines Y

## Our approach

We propose the following workflow

INSERT PIC

- Synthesize new spectra
- Preprocess
- Train a DL model

## Results

We accomplished on training set:

|                       | **Benzine** | **Petroleum** | **Neither** |
|-----------------------|-------------|---------------|-------------|
| **Benzine (Truth)**   | 3222        | 66            | 2           |
| **Petroleum (Truth)** | 0           | 3288          | 10          |
| **Neither (Truth)**   | 1           | 34            | 3377        |


We accomplished on test set:

|                       | **Benzine** | **Petroleum** | **Neither** |
|-----------------------|-------------|---------------|-------------|
| **Benzine (Truth)**   | 29          | 0             | 3           |
| **Petroleum (Truth)** | 0           | 23            | 3           |
| **Neither (Truth)**   | 2           | 0             | 29          |
