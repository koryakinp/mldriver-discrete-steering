import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input("input.jpg", width=7, to='(1,0,0)', height=7),
    to_input("input.jpg", width=7, to='(2,0,0)', height=7),
    to_input("input.jpg", width=7, to='(3,0,0)', height=7),
    to_Conv("conv1", 64, 32, offset="(3,0,0)", to="(5,0,0)", width=10, height=64, depth=64, caption="CONV 1"),
    to_Conv("conv2", 64, 64, offset="(3,0,0)", to="(10,0,0)", width=20, height=48, depth=48, caption="CONV 2"),
    to_Conv("conv3", 64, 64, offset="(0,0,0)", to="(20,0,0)", width=40, height=32, depth=32, caption="CONV 3"),
    to_ConvSoftMax("fc1", 1024, offset="(3,0,0)", to="(30,0,0)", width=5, height=5, depth=300, caption="FC 1"),
    to_ConvSoftMax("fc2", 512, offset="(3,0,0)", to="(35,0,0)", width=5, height=5, depth=250, caption="FC 2"),
    to_ConvSoftMax("fc3", 256, offset="(3,0,0)", to="(40,0,0)", width=5, height=5, depth=200, caption="FC 3"),

    to_SoftMax("actor", 3, offset="(3,0,0)", to="(45,0,5)", width=5, height=5, depth=15, caption="Actor"),
    to_SoftMax("critic", 1, offset="(3,0,0)", to="(45,0,-5)", width=5, height=5, depth=5, caption="Critic"),

    to_connection( "conv1", "conv2"),
    to_connection( "conv2", "conv3"),
    to_connection( "conv3", "fc1"),
    to_connection( "fc1", "fc2"),
    to_connection( "fc2", "fc3"),
    to_connection( "fc3", "actor"),
    to_connection( "fc3", "critic"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()