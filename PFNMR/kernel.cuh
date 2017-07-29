//  PFNMR - Estimate FNMR for proteins (to be updated)
//      Copyright(C) 2016 Jonathan Ellis and Bryan Gantt
//
//  This program is free software : you can redistribute it and / or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation, either version 3 of the License.
//
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//      GNU General Public License for more details.
//
//      You should have received a copy of the GNU General Public License
//      along with this program.If not, see <http://www.gnu.org/licenses/>.

#ifndef __KERNEL_H
#define __KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUTypes.h"

cudaError_t sliceDensityCuda(float *out, const GPUAtom *inAtoms, const GridPoint *inGrid,
    const float variance, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp);

cudaError_t sliceDielectricCuda(float *out, const float *in, const float refDielectric,
    const float outdielectric, const size_t nAtoms, const size_t nGridPoints, cudaDeviceProp &deviceProp);

cudaError_t eFieldDensityCuda(float *out, float *xspans, const GPUChargeAtom *inAtoms, const GPUEFP efp,
    const float variance, const size_t offset, const size_t resopsperiter, const size_t nAtoms,
    const size_t resolution, cudaDeviceProp &deviceProp);

cudaError_t eFieldDielectricCuda(float *out, const float *inDensity, const float innerdielectric,
    const float outerdielectric, const size_t offset, const size_t resopsperiter, const size_t nAtoms,
    const size_t resolution, cudaDeviceProp &deviceProp);

cudaError_t trapIntegrationCuda(float *out, const float *inXSpans, const float *inY, const size_t nStrips,
    const size_t nPoints, cudaDeviceProp &deviceProp);

cudaError_t sqrtf2DCuda(float *out, const size_t nX, const size_t nY, cudaDeviceProp &deviceProp);

cudaError_t electricFieldComponentCuda(GPUEFP *out, const float *inEffLengths, const GPUChargeAtom *inAtoms,
    const float coulconst, const size_t nEFPs, const size_t nAtoms, cudaDeviceProp &deviceProp);

cudaError_t electricPotentialCuda(float *out, const float *inEffLengths, const GPUChargeAtom *inAtoms,
    const float coulconst, const size_t nEFPs, const size_t nAtoms, cudaDeviceProp &deviceProp);

cudaError_t eFieldDensityGQCuda(float *out, float *xspans, const GPUChargeAtom *inAtoms, const float *inAbsci, const GPUEFP efp,
    const float variance, const size_t offset, const size_t resopsperiter, const size_t nAtoms,
    const size_t resolution, cudaDeviceProp &deviceProp);

cudaError_t gaussQuadIntegrationCuda(float *out, const float *inXSpans, const float *inY, const float *inWeights, const size_t nStrips,
    const size_t nPoints, cudaDeviceProp &deviceProp);

#endif
