from termcolor import cprint

class SPrint:
    def __init__(self):
        self.active = True
        self.color = {
            1: 'green',
            2: 'yellow',
            3: 'blue',
            4: 'red',
        }

    def p(self, stringa, indent):
        if self.active:
            stringa = '@@@ ' + '- ' * indent + stringa

            if indent == 1:
                stringa = '\n' + stringa

            cprint(stringa, self.color[indent])

    def deactivate(self):
        self.active = False

    def activate(self):
        self.active = True
