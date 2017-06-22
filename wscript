
import sys

top='.'
out='.build'

tools = [
    'compiler_cxx',
    'cuda',
    'opencv',
]

def options(opt):
    opt.load(tools)

def configure(conf):
    conf.load(tools)

    if sys.platform == 'win32':
        conf.env.CXXFLAGS=[
            # '/WX',
            # '/W4',
            '/EHsc',
            '/nologo',
            '/O2',
        ]

        conf.env.CUDAFLAGS=[
            '-O3',
            '-arch=compute_50',
            '--cl-version=2015',
            '--use-local-env',
        ]
    else:
        conf.env.CXXFLAGS=[
            '-O3',
            '-Wall',
            '-Wextra',
            '-m64',
        ]

        conf.env.CUDAFLAGS=[
            '-O3',
            '-arch=sm_30',
            '-Xcompiler',
            '-Wall',
            '-Xcompiler',
            '-Wextra',
            '-m64',
        ]

def build(bld):
    bld.load(tools)
    bld.recurse('src/ps1')
    bld.recurse('src/ps2')
    bld.recurse('src/ps3')
