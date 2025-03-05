"""
Tkinter UI to compare inference speeds concurrently
"""
import multiprocessing
from tkinter import ttk
import tkinter


def main():
    tk_root = tkinter.Tk()
    frame = ttk.Frame(tk_root, padding=10)
    frame.grid()
    label = ttk.Label(frame, text="Hello World!")
    label.grid(column=0, row=0)
    button = ttk.Button(frame, text="Quit", command=tk_root.destroy)
    button.grid(column=1, row=0)
    tk_root.mainloop()


if __name__ == '__main__':
    main()