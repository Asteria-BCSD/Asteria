# encoding=utf-8
'''

This is a ida plugin script
## 1
To dump all/one AST feature(s) of function(s) to a database file.
## 2
To Calculate Similarity between all functions and functions in another database file.
'''

import idaapi
from idaapi import IDA_SDK_VERSION, IDAViewWrapper
import idc
import idautils
import logging
from ida_kernwin import Form, PluginForm
import os, sys
import json
import commands
import subprocess
import time

try:
    if IDA_SDK_VERSION < 690:
        # In versions prior to IDA 6.9 PySide is used...
        from PySide import QtGui

        QtWidgets = QtGui
        is_pyqt5 = False
    else:
        # ...while in IDA 6.9, they switched to PyQt5
        from PyQt5 import QtCore, QtGui, QtWidgets

        is_pyqt5 = True
except ImportError:
    pass

l = logging.getLogger("ast_generator.py")
l.setLevel(logging.ERROR)
logger = logging.getLogger('Asteria')
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler("Asteria.log"))
logger.handlers[0].setFormatter(
    logging.Formatter("[%(filename)s][%(levelname)s] %(message)s\t(%(module)s:%(funcName)s)"))
logger.handlers[1].setFormatter(
    logging.Formatter("[%(filename)s][%(levelname)s] %(message)s\t(%(module)s:%(funcName)s)"))
logger.setLevel(logging.ERROR)


# logging.basicConfig(format='[%(filename)s][%(levelname)s] %(message)s\t(%(module)s:%(funcName)s)',
#                     filename="Asteria.log",
#                     filemode='a')

# view to show the similaity results
class CHtmlViewer(PluginForm):

    def OnCreate(self, form):
        if is_pyqt5:
            self.parent = self.FormToPyQtWidget(form)
        else:
            self.parent = self.FormToPySideWidget(form)
        self.PopulateForm()

        self.browser = None
        self.layout = None
        self.text = ""

        return 1

    def PopulateForm(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.browser = QtWidgets.QTextBrowser()
        self.browser.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.browser.setHtml(self.text)
        self.browser.setReadOnly(True)
        self.browser.setFontWeight(18)
        self.layout.addWidget(self.browser)
        self.parent.setLayout(self.layout)

    def ScoreToColor(self, score):
        '''
        return the color rgb value according to score
        1. score ranges 0.8~1.0
        2. color format #ff{A}{A} , A is calculated according to Score
        '''
        score = float(score)
        if score < 0.8:
            return "#ffffff"  # white

        ColorScore = int(125 - (score - 0.8) * 10 * 62)
        return "#ff%.2x%.2x" % (ColorScore, ColorScore)

    def AssembleHtml(self, jsondata):
        '''
        jsondata :dic: {'vulFuncName':{'info':['funcName','BinPath','BinName'], "rank":[[[like info], score], [[like info], score]]}}
        '''
        _html_template = """
                              <html>
                              <head>
                              <style>%(style)s</style>
                              </head>
                              <body>
                              <table class="diff_tab"  cellspacing="3px">
                              %(rows)s
                              </table>
                              </body>
                              </html>
        """
        _style = """
                 table.diff_tab {
                    font-family: Courier monospace;
                    border: solid black;
                    font-size: 15px;

                    text-align: left;
                    width: 100%;
                  }
                  td {
                    border: 2px;
                    border-bottom-style:solid;
                  }
                  """

        head_row_temp = """
                    <tr> 
                    <td class="VulFuncName" rowspan="%(rowNumber)d"> %(vulnerableFunc)s </td>
                    <td class="CandidateName"> %(candidateFunc)s </td>
                    <td class="score" bgcolor=%(bgcolor)s> %(score)s </td>
                    </tr>
        """
        other_row_temp = """
                    <tr> 
                    <td > %(candidateFunc)s </td>
                    <td bgcolor=%(bgcolor)s> %(score)s </td>
                    </tr>
        """
        table_header = """
            <thead> 
                <tr> 
                <th>Functions In Chosen File</th> <th>Functions In IDA</th>  <th>Similarity Scores</th> 
                </tr> 
            </thead>
        """
        table_content = ""
        for vulFunc in jsondata:
            candidates = jsondata[vulFunc]['rank']
            rowNumber = len(candidates)
            if rowNumber == 0:
                continue
            vulInfo = jsondata[vulFunc]['info']
            vulFuncName = vulFunc
            vulFuncBin = vulInfo[2]

            headerRow = head_row_temp % {"rowNumber": rowNumber, "vulnerableFunc": vulFuncName,
                                         "candidateFunc": candidates[0][0][0], "score": candidates[0][1],
                                         "bgcolor": self.ScoreToColor(candidates[0][1])}
            table_content += headerRow
            if rowNumber > 1:
                for candidate in candidates[1:]:
                    table_content += other_row_temp % {"candidateFunc": candidate[0][0], "score": candidate[1],
                                                       "bgcolor": self.ScoreToColor(candidate[1])}

        src = _html_template % {"style": _style, "rows": table_header + table_content}
        logger.debug(src)
        return src

    def Show(self, jsondata, title):
        self.text = self.AssembleHtml(jsondata)
        """Creates the form is not created or focuses it if it was"""
        return PluginForm.Show(self, title, options=PluginForm.FORM_PERSIST)


# function menus
class MyForm(Form):
    # Main Class for functionality of plugin
    def __init__(self):
        self.invert = False
        F = ida_kernwin.Form
        F.__init__(
            self,
            r"""STARTITEM 0
            Please choose your option.
            <##Function Feature Genration:{iButton1}> <##Function Similarity:{iButton2}>
            """,
            {
                'iButton1': F.ButtonInput(self.db_generate),
                'iButton2': F.ButtonInput(self.sim_cal)
            }
        )
        self._sqlitefilepath = idc.GetIdbPath() + ".sqlite"
        idadir = idaapi.idadir("python")
        self.mainapp_path = os.path.join(idadir, "ASTExtraction",
                                         "main_app.py")  # path to the script of the ast encoding and similarity
        if not os.path.exists(self.mainapp_path):
            logger.error("Python File {} not exists!".format(self.mainapp_path))
        self.application = os.path.join(idadir, "ASTExtraction",
                                        "application.py")  # path to the script of the ast encoding and similarity
        if not os.path.exists(self.application):
            logger.error("Python File {} not exists!".format(self.application))

    def db_generate(self, code=0):
        logger.debug("ast generating...")
        # 1. set file to save; 2. show waiting box and progress 3. invoke ida script for ast generation 4. show sucess box.
        # 1.
        logger.debug("sqlite file path {}".format(self._sqlitefilepath))

        # 2.
        ida_kernwin.show_wait_box("Processing...")

        # 3.
        # 3.1 extract asts
        try:
            idaapi.require("ASTExtraction")
            idaapi.require("ASTExtraction.ast_generator")
            idaapi.require("ASTExtraction.DbOp")
            g = ASTExtraction.ast_generator.AstGenerator()
            dbop = ASTExtraction.DbOp.DBOP(self._sqlitefilepath)
            g.run(g.get_info_of_func)
            # 3.2 save asts to database
            ida_kernwin.replace_wait_box("Processing. Saving to Database.")
            recordsNo = g.save_to(dbop)
            logger.info("%d records are inserted into database." % recordsNo)
            del dbop
        except Exception, e:
            ida_kernwin.replace_wait_box(str(e))
            logger.error("ASTExtraction import error! {}".format(e))
            time.sleep(2)
        # 4.
        ida_kernwin.replace_wait_box("AST Encoding...")

        cmd = 'python "{}" --dbpath "{}"'.format(self.application, self._sqlitefilepath)
        idaapi.msg("[Asteria] >>> AST Encoding...[{}]\n".format(cmd))

        returncode = self.invoke_system_python(cmd, hidden_window=False)
        if returncode:
            idaapi.msg("[Asteria] >>> AST Encoding failed\n")
        ida_kernwin.hide_wait_box()
        idaapi.msg("[Asteria] >>> AST Encoding Finished\n")

    def invoke_system_python(self, cmd, hidden_window=True):
        '''
        cmd : str: command to be executed
        hidden_window: bool:
        return : subprocess.returncode
        '''
        startupinfo = subprocess.STARTUPINFO()
        # to hidden the cmd window
        if hidden_window:
            if 'win32' in str(sys.platform).lower():
                startupinfo.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            p = subprocess.Popen(cmd, startupinfo=startupinfo,
                                 stdout=subprocess.PIPE)  # , stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            while p.poll() == None:
                idaapi.msg(p.stdout.read(1))
            return p.returncode

        p = subprocess.Popen(cmd,
                             startupinfo=startupinfo)  # , stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        # while p.poll() == None:
        #     idaapi.msg(p.stdout.readline().strip())
        # p.stdout.flush()
        return p.wait()

    def sim_cal(self, code=0):
        logger.debug("Similarity Calculation...")
        # : 1. make sure current idb has generated sqlite db file 2. Choose sqlite db file to compare with 3. do calculation 4. show similarity results.
        # 1.
        if os.path.exists(self._sqlitefilepath):
            # 2.
            sqlite_database2, _ = QtWidgets.QFileDialog.getOpenFileName()  # another sqlite db file for similarity calculation with the sqlite db file generated by current idb
            idaapi.msg("[Asteria] >>> The sqlite file to be calculated is {}\n".format(sqlite_database2))
            # 3. 3.1: TODO: should check the python environment first
            # 3.2: do the calculation and save the results to file

            result_path = self._sqlitefilepath.split(".")[0] + ".json"
            cmd = 'python "{}" --result "{}" "{}" "{}"'.format(self.mainapp_path, result_path, sqlite_database2,
                                                               self._sqlitefilepath)
            logger.info("Calculation CMD: {}".format(cmd))
            ida_kernwin.show_wait_box("Calculating... Please wait.")
            returncode = self.invoke_system_python(cmd, hidden_window=True)
            ida_kernwin.hide_wait_box()
            if returncode:
                # cmd execution fails
                idaapi.msg("[Asteria] >>> Calcualtion failed, {}.\n".format(returncode))
            # 3.3 load the results
            if not os.path.exists(result_path):
                idaapi.msg("[Asteria] >>> Result file does not exists after the calculation, please see the log.\n")
                return

            results = json.load(open(result_path, 'r'))
            # logger.info(str(results))
            # show window of result
            window = CHtmlViewer()
            title = "Similarity Scores"
            window.Show(results, title)
            self.Close(1)
            # self.Free()
        else:
            ida_kernwin.show_wait_box("Please run AST Generation first.")
            while not ida_kernwin.user_cancelled():
                pass
            ida_kernwin.hide_wait_box()
            return


class AsteriaPlugin(idaapi.plugin_t):
    """
    This is the main class of the plugin. It subclasses plugin_t as required
    by IDA. It holds the modules of plugin, which themselves provides the
    functionality of the plugin.
    """
    # plugin information
    flags = idaapi.PLUGIN_UNL
    comment = "This is a plugin for the Asteria (https://github.com/Asteria-BCSD/Asteria)"
    wanted_name = "Asteria"  # name of plugin
    wanted_hotkey = "Alt-F1"  # hot key
    help = "Please see README in https://github.com/Asteria-BCSD/Asteria"

    def init(self):
        if not self.load_plugin_decompiler():
            idaapi.msg("[Asteria]Failed to load hexray plugin.")
            return idaapi.PLUGIN_SKIP

        # To load the modules in /path/to/ida/python/Asteria

        try:
            idaapi.require("ASTExtraction")
            # idaapi.require("ASTExtraction.ast_generator")
            # idaapi.require("ASTExtraction.DbOp")
        except Exception, e:
            idaapi.msg(
                "[Asteria] Plugin initialization failed. Please read the README file and copy 'ASTExtraction' dir of the project to '/path/to/ida/python/'\n")
            logger.error(e.args)
            return idaapi.PLUGIN_SKIP

        idaapi.msg("[Asteria] >>> Asteria plugin initialization finished!\n")
        return idaapi.PLUGIN_OK  # return PLUGIN_KEEP

    def run(self, arg):
        self.form = MyForm()
        self.form.Compile()
        self.form.Execute()
        logger.debug(">>> Asteria plugin exists")

    def term(self):
        idaapi.msg("[Asteria] >>> Asteria plugin ends.\n")

    def load_plugin_decompiler(self):
        '''
        load the hexray plugins
        :return: success or not
        '''
        is_ida64 = idc.GetIdbPath().endswith(".i64")
        if not is_ida64:
            idaapi.load_plugin("hexrays")
            idaapi.load_plugin("hexarm")
        else:
            idaapi.load_plugin("hexx64")
        if not idaapi.init_hexrays_plugin():
            logger.error('[+] decompiler plugins load failed. IDAdb: %s' % idc.GetInputFilePath())
            return False
        return True


def PLUGIN_ENTRY():
    # this function returns an instance of an idapython plugin

    return AsteriaPlugin()
