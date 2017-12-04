
import subprocess


for i in range(1,8):

    print("convert liver_{}\n".format(i))

    p = subprocess.Popen(["/home/staff/reichard/workspace/mediassist-build/bin/endoCheck",
                             "/org/share/MediData/MedData/Tier/Leber/26-02-14/Aufnahme{}/".format(i),
                             "/local_home/daniel/sparse_pig/liver{}/".format(i)], stdout=subprocess.PIPE)

    while p.poll() is None:
        l = p.stdout.readline() # This blocks until it receives a newline.
        print(l)
    # When the subprocess terminates there might be unconsumed output
    # that still needs to be processed.
    print(p.stdout.read())