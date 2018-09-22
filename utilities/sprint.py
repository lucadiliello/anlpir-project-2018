from termcolor import cprint

def p(stringa, indent):
    color = {
        1: 'green',
        2: 'yellow',
        3: 'blue',
        4: 'red',
    }
    stringa = '@@@ ' + '- ' * indent + stringa
    if indent == 1:
        stringa = '\n' + stringa

    cprint(stringa, color[indent])
