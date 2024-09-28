import atexit

def exit():
    print('CNTRL C PRESSED')
atexit.register(exit)
input()