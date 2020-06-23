

You just need to use the `API_ast_generator.py` to extract ASTs.

# usage
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