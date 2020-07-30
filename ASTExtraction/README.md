
# command line usage
## 1. You just need to use the `API_ast_generator.py` to extract ASTs.

### usage help
```
usage: API_ast_generator.py [-h] [--ida_path IDA_PATH] [--binary BINARY]
                            [--directory DIRECTORY] [--database DATABASE]
                            [--logfile LOGFILE] [--function FUNCTION]
                            [--compilation {default,O0,O1,O2,O3,Os}]
                            [--timeout TIMEOUT]

AST generator script based on IDA and  Python3 !

optional arguments:
  -h, --help            show this help message and exit
  --ida_path IDA_PATH   path to idal(before 7.2) or idat(after 7.2) with
                        decompiler plugins, idal64 for 64bit binary(also for
                        32bit)
  --binary BINARY       path to binary to be analysed
  --directory DIRECTORY
                        path, all binaries in dir will be analysed. --binary
                        will not work if specified
  --database DATABASE   path to sqlite database to save the extracted asts
  --logfile LOGFILE     log file when ida runs
  --function FUNCTION   specific function name, default is to get all function
                        ast
  --compilation {default,O0,O1,O2,O3,Os}
                        specify the compilation level of binary
  --timeout TIMEOUT     max seconds a binary cost
```
## 2. use `application.py` encodes the asts of generated database.

# Use as IDA Plugin

## install
1. copy `Asteria_ida_plugin.py` to `/path/to/ida/plugin`. `/path/to/ida/plugin` is the plugin directory in your ida installation directory.
2. copy the directory `ASTExtraction` to `/path/to/ida/python/`
3. make sure the python libraries `numpy, torch, gevent, sklearn` have been installed in your local python environment

## usage
1. Use shortcut key `ALT+F1` or `Edit->Plugin->Asteria` to invoke the main menu

2. To dump all/one AST feature(s) of function(s) to a database file with button `Function Feature Generation`.

3. To Calculate Similarity between all functions and functions in another database file with button `Function Similairty`.
    * Choose the sqlite file in which the **ast encodes have been generated**.
   aa
[Result](./SimRes.PNG)
    
