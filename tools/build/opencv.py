#!/usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2010

import os

def configure(conf):
    try:
        d = conf.root.find_node(os.environ['OPENCV_BASE_PATH'])

        node = d.find_node('include')
        _includes = node.abspath()

        node = d.find_node('x64').find_node('vc14').find_node('lib')
        _libpath = [node.abspath()]

        conf.check_cxx(lib='opencv_world320', libpath=_libpath, includes=_includes, uselib_store='OPENCV')
    except KeyError:
        conf.fatal('OPENCV_BASE_PATH environment variable not defined')

