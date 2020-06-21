# Asteria

This is the official repository for Asteria.


## Files description

* train.py: python file for model training
* Tree.py: The Tree class
* datahelper.py: function set for accessing sqlite database
## dirs description
* application: prototype for Asteria
* data: sqlite database files of ASTs
* diaphora_test: the implement of BCSD based AST of Diaphora


# Functions
1. AST extraction and preprocessing
2. model training
3. vulnerability search

# Usage

## Vulnerability Search
`python application/main_app.py "database of vulnerable functions" "database of firmware functions"
For example:
 `python application/main_app.py data/vul.sqlite data/NetGear_Small.sqlite`
 
 Then the search results are saved into "VulSearch.result".
 
 The format of results is as follows:
 ```
[VULFUNC]:   ftp_retrieve_glob  VULELF:/home/ubuntu/disk/hdd_1/ysg/binary_pool/vulnerable_set/CVE-2014-4877/wget
        |Sim:1.000      |Func:   ftp_retrieve_glob      |ELFPath:/home/ubuntu/disk/hdd_2/iie/acfg/firmwareExtracted/NetGear/_R8000-V1.0.3.32_1.1.21.zip.extracted/_R8000-V1.0.3.32_1.1.21.chk.extracted/squashfs-root/bin/wget

        |Sim:1.000      |Func:   ftp_retrieve_glob      |ELFPath:/home/ubuntu/disk/hdd_2/iie/acfg/firmwareExtracted/NetGear/_R8000-V1.0.2.44_1.0.96.chk.extracted/squashfs-root/bin/wget

```

The VULFUNC denotes the name of vulnerable function. The Sim denotes the similarity score. The Func denotes the candidate function name.
The ELFPath denotes the path of binary where candidate function come from.