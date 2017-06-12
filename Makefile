CXX = clang++
CXXFLAGS = -Wall -g -O3 -std=c++11
OBJS = mnist.o data.o network.o
HDRS = data.hpp network.hpp

all:	mnist

mnist:	${OBJS}
	${CXX} ${CXXFLAGS} -o $@.out ${OBJS}

.o:	${HDRS}

clean:
	${RM} ${OBJS}
