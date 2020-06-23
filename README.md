# Asteria

This is the official repository for Asteria.


## Files description

* train.py: python file for model training
* Tree.py: The Tree class defination
* datahelper.py: function set for accessing sqlite database
## dirs description
* application: prototype for Asteria
* data: sqlite database files of ASTs
* diaphora_test: the implement of BCSD based AST of Diaphora
* ASTExtraction: scripts for AST extraction.

# Functions
1. AST extraction and preprocessing
2. model training
3. vulnerability search

# Usage

## Vulnerability Search
`python application/main_app.py "database of vulnerable functions" "database of firmware functions`
### For example:
 `python application/main_app.py data/vul.sqlite data/NetGear_Small.sqlite`
 
 vul.sqlite saves the ASTs of vulnerable functions.
 NetGear_Small.sqlite saves the ASTs of part of firmware functions from NetGear manufacture.
 
 Then the search results are saved into "VulSearch.result".
 
 The format of results is as follows:
 ```
[VULFUNC]:   ftp_retrieve_glob  VULELF:/home/ubuntu/disk/hdd_1/ysg/binary_pool/vulnerable_set/CVE-2014-4877/wget
        |Sim:1.000      |Func:   ftp_retrieve_glob      |ELFPath:/home/ubuntu/disk/hdd_2/iie/acfg/firmwareExtracted/NetGear/_R8000-V1.0.3.32_1.1.21.zip.extracted/_R8000-V1.0.3.32_1.1.21.chk.extracted/squashfs-root/bin/wget

        |Sim:1.000      |Func:   ftp_retrieve_glob      |ELFPath:/home/ubuntu/disk/hdd_2/iie/acfg/firmwareExtracted/NetGear/_R8000-V1.0.2.44_1.0.96.chk.extracted/squashfs-root/bin/wget

```

The VULFUNC denotes the name of vulnerable function. The Sim denotes the similarity score. The Func denotes the candidate function name.
The ELFPath denotes the path of binary where candidate function come from.


## Model Training
Since the buildroot dataset we used is too large(28G), we construct a demo training dataset for demonstrating.

`python train.py`
After 60 epochs training, the model with best performance is saved in "checkpoints/crossarch.pt".

The trained model parameters and settings in our work are placed in "data/saved_model.pt"