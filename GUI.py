import PySimpleGUI as sg
import subprocess
import sys

def runCommand(cmd, timeout=None, window=None):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ''
    for line in p.stdout:
        line = line.decode(errors='replace' if (sys.version_info) < (3, 5) else 'backslashreplace').rstrip()
        output += line
        print(line)
        window.Refresh() if window else None        # yes, a 1-line if, so shoot me
    retval = p.wait(timeout)
    return (retval, output)

sg.theme('DarkGrey14')

layout = [
    [sg.Text('enter command....', size=(40, 1))],
    [sg.Output(size=(88, 20), font='Courier 10')],
    [sg.Button('EXIT')],
    [sg.Text('Manual command', size=(15, 1)), sg.Input(focus=True, key='-IN-'), sg.Button('Run', bind_return_key=True)]
]

window = sg.Window('Script launcher', layout)

# ---===--- Loop taking in user input and using it to call scripts --- #
while True:             # Event Loop
        event, values = window.Read()
        if event in (None, 'EXIT'):         # checks if user wants to exit
            print("!!!")
            break

        if event == 'Run':                  # the two lines of code needed to get button and run command
            runCommand(cmd=values['_IN_'], window=None)

window.Close()





# def main():
#
#     layout = [  [sg.Text('Enter a command to execute (e.g. dir or ls)')],
#                 [sg.Input(key='_IN_')],             # input field where you'll type command
#                 [sg.Output(size=(100,40))],          # an output area where all print output will go
#                 [sg.Button('Run'), sg.Button('Exit')]]    # a couple of buttons
#
#     window = sg.Window('Realtime Shell Command Output', layout)
#
#     while True:             # Event Loop
#         event, values = window.Read()
#         if event in (None, 'Exit'):         # checks if user wants to exit
#             break
#         if event == 'Run':                  # the two lines of code needed to get button and run command
#             runCommand(cmd=values['_IN_'], window=None)
#
#     window.Close()
#
#
# #This function does the actual "running" of the command.  Also watches for any output. If found output is printed

#
# if __name__ == '__main__':
#     main()
#
#







# import PySimpleGUI as sg
#
# # Define the window's contents
# layout = [[sg.Text("What's your name?")],
#           [sg.Input(key='-INPUT-')],
#           [sg.Text(size=(100,50), key='-OUTPUT-')],
#           [sg.Button('Ok'), sg.Button('Quit')]]
#
# # Create the window
# window = sg.Window('Window Title', layout)
#
# # Display and interact with the Window using an Event Loop
# while True:
#     event, values = window.read()
#     # See if user wants to quit or window was closed
#     if event == sg.WINDOW_CLOSED or event == 'Quit':
#         break
#     # Output a message to the window
#     window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] + "! Thanks for trying PySimpleGUI")
#
# # Finish up by removing from the screen
# window.close()