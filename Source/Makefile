VPATH = Array Complex DSP Matrix Operations Tensor Types
SDK=$$(xcrun --show-sdk-path --sdk macosx)
SRC=ValueArray.swift ValueArraySlice.swift DSP.swift Matrix.swift MatrixArithmetic.swift Tensor.swift TensorSlice.swift Span.swift Arithmetic.swift Auxiliary.swift Exponential.swift Hyperbolic.swift Trigonometric.swift ContiguousMemory.swift Interval.swift Real.swift Value.swift 
OBJS:=$(subst .swift,.o,$(SRC))

.PHONY: all
all: libUpsurge.a Upsurge.swiftmodule

libUpsurge.a: $(OBJS)
	libtool -static -s -o $@ -no_warning_for_no_symbols $^

Upsurge.swiftmodule: $(SRC)
	xcrun swiftc -emit-module $^ -sdk $(SDK) -module-name Upsurge

$(OBJS): $(SRC)
	xcrun swiftc -emit-library -emit-object $^ -sdk $(SDK) -module-name Upsurge

.PHONY: clean
clean:
	rm -f *.o *.a *.swiftdoc *.swiftmodule
