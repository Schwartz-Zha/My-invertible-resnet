from os import listdir
from os.path import isfile, join


def snippet_test():
    mypath = 'models/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    pythonfiles = [f for f in onlyfiles if f.endswith('.py')]
    print(onlyfiles)
    print(pythonfiles)

if __name__ == '__main__':
    snippet_test()