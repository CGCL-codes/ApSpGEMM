#pragma once

#include "dCSR.h"

namespace ApSpGEMM {
	template <typename DataType>
	void Transpose(const dCSR<DataType>& matIn, dCSR<DataType>& matTransposeOut);
}