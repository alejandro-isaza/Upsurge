SDK=$$(xcrun --show-sdk-path --sdk macosx)
SWIFT=/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc
OBJS=ValueArray.o ValueArraySlice.o Matrix.o MatrixArithmetic.o Tensor.o TensorSlice.o Span.o Arithmetic.o Auxiliary.o Exponential.o ContiguousMemory.o Interval.o Real.o Value.o
SRC=./Array/ValueArray.swift ./Array/ValueArraySlice.swift ./Matrix/Matrix.swift ./Matrix/MatrixArithmetic.swift ./Tensor/Tensor.swift ./Tensor/TensorSlice.swift ./Tensor/Span.swift ./Operations/Arithmetic.swift ./Operations/Auxiliary.swift ./Operations/Exponential.swift ./Types/ContiguousMemory.swift ./Types/Interval.swift ./Types/Real.swift ./Types/Value.swift 
VPATH = Array

.PHONY: all
all: lib module

lib: $(OBJS)
	libtool -static -s -o libUpsurge.a -no_warning_for_no_symbols $(OBJS)

module:
	xcrun swiftc -emit-module $(SRC) -sdk $(SDK) -module-name Upsurge

$(OBJS): $(SRC)
	xcrun swiftc -emit-library -emit-object $(SRC) -sdk $(SDK) -module-name Upsurge

#ValueArray.o: ValueArray.swift
#	$(SWIFT) -c -primary-file $< $(SRC) -sdk $(SDK) -module-name Upsurge -o $@

.PHONY: clean
clean:
	rm -f *.o *.a *.swiftdoc *.swiftmodule
