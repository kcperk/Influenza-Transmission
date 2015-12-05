all: main.cu
        nvcc -arch=sm_21 -ccbin=gcc -Xcompiler -Wshadow -o infection $^

clean:
        rm -f infection
