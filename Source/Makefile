SDK=$$(xcrun --show-sdk-path --sdk macosx)
OBJS=Value.o Real.o ValueArray.o ValueArraySlice.o ContiguousMemory.o Matrix.o
SRC=./Types/Value.swift ./Array/ValueArray.swift ./Array/ValueArraySlice.swift ./Types/ContiguousMemory.swift ./Matrix/Matrix.swift ./Types/Real.swift ./Tensor/Tensor.swift ./Tensor/TensorSlice.swift ./Tensor/Span.swift ./Types/Interval.swift

.PHONY: all
all: lib module

lib: $(OBJS)
	libtool -static -s -o libUpsurge.a -no_warning_for_no_symbols $(OBJS)

module:
	xcrun swiftc -emit-module $(SRC) -sdk $(SDK) -module-name Upsurge

$(OBJS): $(SRC)
	xcrun swiftc -emit-library -emit-object $(SRC) -sdk $(SDK) -module-name Upsurge

.PHONY: clean
clean:
	rm -f *.o *.a *.swiftdoc *.swiftmodule
