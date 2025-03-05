"""
Tkinter UI to compare inference speeds concurrently
"""
import multiprocessing
from threading import Thread
from tkinter.ttk import Frame, Label, Button
import tkinter
import time


class App(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.am_i_alive = False
        self.tk_root = None
        self.frame = None
        self.left_frame = None
        self.left_label = None
        self.right_label = None
        self.right_frame = None
        self.label = None
        self.button = None

    def ui_setup(self):
        self.tk_root = tkinter.Tk()
        self.frame = Frame(self.tk_root, padding=10)
        self.frame.grid()
        self.label = Label(self.frame, text="Hello World!")
        self.label.grid(column=0, row=0)
        self.button = Button(self.frame, text="Quit", command=self.tk_root.destroy)
        self.button.grid(column=1, row=0)

        self.left_frame = Frame(self.frame)
        self.left_frame.grid(column=0, row=1)
        self.left_frame.configure(height=600, width=600)
        self.left_label = Label(self.frame, text="CPU Inference")
        self.left_label.grid(column=0, row=2)

        self.right_frame = Frame(self.frame)
        self.right_frame.grid(column=1, row=1)
        self.right_frame.configure(height=600, width=600)
        self.left_label = Label(self.frame, text="NNPA Inference")
        self.left_label.grid(column=1, row=2)

    def run(self) -> None:
        self.ui_setup()
        self.am_i_alive = True
        self.tk_root.mainloop()
        self.am_i_alive = False

    def is_alive(self):
        """ Returns true while the UI is alive. False when it ends. """
        return self.am_i_alive

def main():
    app = App()
    app.start()
    while True:
        if not app.is_alive():
            break
        time.sleep(1)


if __name__ == '__main__':
    main()